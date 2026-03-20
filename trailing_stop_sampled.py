import blpapi
import csv
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


# =========================================================
# USER SETTINGS
# =========================================================
SECURITY = "USDCAD Curncy"

# "long"  -> trailing sell stop, monitors BID
# "short" -> trailing buy stop, monitors ASK
SIDE = "long"

FIELD = "BID" if SIDE.lower() == "long" else "ASK"

HOST = "localhost"
PORT = 8194

# Example: 100 pips in USDCAD terms = 0.0100
STOP_DISTANCE = 0.0100

# Set explicitly, or leave as None to initialise from the first sampled tick
ENTRY_PRICE: Optional[float] = 1.3500

# Sampling interval in seconds
SAMPLE_SECONDS = 1.0

# Output file
CSV_FILE = "trailing_stop_sampled.csv"

# If True, skip logging when the sampled price hasn't changed since the last sample
LOG_ONLY_IF_SAMPLED_PRICE_CHANGED = True

# Max Bloomberg session-start retries before giving up
MAX_SESSION_RETRIES = 5
SESSION_RETRY_DELAY = 5.0   # seconds between retries


# =========================================================
# DATA STRUCTURES
# =========================================================
@dataclass
class TickState:
    timestamp: Optional[datetime] = None
    price: Optional[float] = None


@dataclass
class EngineState:
    security: str
    side: str
    field: str
    stop_distance: float
    entry_price: Optional[float] = None
    trailing_stop: Optional[float] = None
    stop_triggered: bool = False
    trigger_timestamp: Optional[str] = None
    trigger_price: Optional[float] = None


# =========================================================
# ENGINE
# =========================================================
class TrailingStopEngine:
    def __init__(self, state: EngineState):
        self.state = state

    def initialize_if_needed(self, first_price: float):
        """Initialise entry price and trailing stop from the first sampled tick
        (only called when ENTRY_PRICE was left as None)."""
        if self.state.entry_price is None:
            self.state.entry_price = first_price

        if self.state.trailing_stop is None:
            if self.state.side == "long":
                self.state.trailing_stop = self.state.entry_price - self.state.stop_distance
            elif self.state.side == "short":
                self.state.trailing_stop = self.state.entry_price + self.state.stop_distance
            else:
                raise ValueError("SIDE must be 'long' or 'short'")

    def update(self, price: float, ts: datetime) -> bool:
        """
        Apply one sampled price to the engine.
        Returns True if the stop was *just* triggered by this tick.
        """
        self.initialize_if_needed(price)

        if self.state.stop_triggered:
            return False  # already done — caller should stop feeding ticks

        if self.state.side == "long":
            candidate = price - self.state.stop_distance
            self.state.trailing_stop = max(self.state.trailing_stop, candidate)

            if price <= self.state.trailing_stop:
                self.state.stop_triggered = True
                self.state.trigger_timestamp = ts.isoformat(timespec="seconds")
                self.state.trigger_price = price
                return True

        elif self.state.side == "short":
            candidate = price + self.state.stop_distance
            self.state.trailing_stop = min(self.state.trailing_stop, candidate)

            if price >= self.state.trailing_stop:
                self.state.stop_triggered = True
                self.state.trigger_timestamp = ts.isoformat(timespec="seconds")
                self.state.trigger_price = price
                return True

        else:
            raise ValueError("SIDE must be 'long' or 'short'")

        return False


# =========================================================
# CSV LOGGER
# =========================================================
class CsvLogger:
    HEADER = [
        "timestamp",
        "security",
        "side",
        "field",
        "sampled_price",
        "entry_price",
        "stop_distance",
        "trailing_stop",
        "stop_triggered",
        "trigger_timestamp",
        "trigger_price",
    ]

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._ensure_header()

    def _ensure_header(self):
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", newline="") as f:
                csv.writer(f).writerow(self.HEADER)

    def write(self, ts: datetime, sampled_price: float, state: EngineState):
        with open(self.filepath, "a", newline="") as f:
            csv.writer(f).writerow([
                ts.isoformat(timespec="seconds"),
                state.security,
                state.side,
                state.field,
                sampled_price,
                state.entry_price,
                state.stop_distance,
                state.trailing_stop,
                state.stop_triggered,
                state.trigger_timestamp,
                state.trigger_price,
            ])


# =========================================================
# MARKET DATA SAMPLER
# =========================================================
class Sampler:
    """
    Buffers incoming ticks and emits one snapshot per interval.
    Uses a single time.time() call per cycle to avoid the
    should_emit / emit race condition.
    """

    def __init__(self, interval_seconds: float):
        self.interval_seconds = interval_seconds
        self._next_emit_at: Optional[float] = None
        self.latest_tick = TickState()
        self.last_sampled_price: Optional[float] = None

    def on_tick(self, price: float, ts: datetime):
        """Store the latest tick; only the most recent matters."""
        self.latest_tick.price = price
        self.latest_tick.timestamp = ts

    def try_emit(self) -> Optional[TickState]:
        """
        Returns a TickState snapshot if the sampling interval has elapsed
        and a tick has been received, otherwise None.
        A single time.time() call ensures no race between check and update.
        """
        if self.latest_tick.price is None or self.latest_tick.timestamp is None:
            return None

        now = time.time()

        if self._next_emit_at is None:
            # First emission: fire immediately
            self._next_emit_at = now + self.interval_seconds
            return TickState(
                timestamp=self.latest_tick.timestamp,
                price=self.latest_tick.price,
            )

        if now >= self._next_emit_at:
            self._next_emit_at = now + self.interval_seconds
            return TickState(
                timestamp=self.latest_tick.timestamp,
                price=self.latest_tick.price,
            )

        return None


# =========================================================
# BLOOMBERG SUBSCRIPTION HANDLER
# =========================================================
class LiveTrailingStopApp:
    def __init__(self):
        self.engine = TrailingStopEngine(
            EngineState(
                security=SECURITY,
                side=SIDE.lower(),
                field=FIELD,
                stop_distance=STOP_DISTANCE,
                entry_price=ENTRY_PRICE,
                trailing_stop=None,
            )
        )
        self.logger = CsvLogger(CSV_FILE)
        self.sampler = Sampler(SAMPLE_SECONDS)
        self._stop_requested = False          # set to True to exit the main loop

        # Cache blpapi Name objects (avoid repeated allocation)
        self._NAME_SUB_DATA       = blpapi.Name("SubscriptionData")
        self._NAME_SUB_STARTED    = blpapi.Name("SubscriptionStarted")
        self._NAME_SUB_FAILURE    = blpapi.Name("SubscriptionFailure")
        self._NAME_SUB_TERMINATED = blpapi.Name("SubscriptionTerminated")

    # ------------------------------------------------------------------
    # Event handler (called by Bloomberg on its own thread)
    # ------------------------------------------------------------------
    def process_event(self, event, session):
        event_type = event.eventType()

        for msg in event:
            msg_type = msg.messageType()

            if msg_type == self._NAME_SUB_STARTED:
                print(f"[STATUS] Subscription started.")
                continue
            if msg_type == self._NAME_SUB_FAILURE:
                print(f"[ERROR]  Subscription failure: {msg}")
                continue
            if msg_type == self._NAME_SUB_TERMINATED:
                print(f"[STATUS] Subscription terminated.")
                self._stop_requested = True
                continue

            if msg_type == self._NAME_SUB_DATA:
                if msg.hasElement(FIELD):
                    price = msg.getElementAsFloat(FIELD)
                    self.sampler.on_tick(price=price, ts=datetime.now())

        # After digesting the event, check whether a sample is due
        self._maybe_emit_sample(session)

    def _maybe_emit_sample(self, session):
        """Emit a sample if the interval has elapsed; handle stop triggering."""
        sampled = self.sampler.try_emit()
        if sampled is None:
            return

        sampled_price = sampled.price
        sampled_ts    = sampled.timestamp

        # Skip unchanged prices if configured to do so
        if LOG_ONLY_IF_SAMPLED_PRICE_CHANGED:
            if (self.sampler.last_sampled_price is not None
                    and sampled_price == self.sampler.last_sampled_price):
                return

        # FIX: stop processing ticks once stop is already triggered
        if self.engine.state.stop_triggered:
            return

        just_triggered = self.engine.update(price=sampled_price, ts=sampled_ts)
        self.logger.write(ts=sampled_ts, sampled_price=sampled_price, state=self.engine.state)
        self.sampler.last_sampled_price = sampled_price

        print(
            f"{sampled_ts.strftime('%H:%M:%S')} | "
            f"{SECURITY} {FIELD}={sampled_price:.5f} | "
            f"Stop={self.engine.state.trailing_stop:.5f} | "
            f"Triggered={self.engine.state.stop_triggered}"
        )

        if just_triggered:
            print(
                f"\n{'='*50}\n"
                f"[TRIGGERED] {self.engine.state.side.upper()} stop hit!\n"
                f"  Time  : {self.engine.state.trigger_timestamp}\n"
                f"  Price : {self.engine.state.trigger_price:.5f}\n"
                f"{'='*50}\n"
            )
            # FIX: cleanly request session teardown after trigger
            self._stop_requested = True

    # ------------------------------------------------------------------
    # Session management with retry logic
    # ------------------------------------------------------------------
    def _build_session(self) -> blpapi.Session:
        options = blpapi.SessionOptions()
        options.setServerHost(HOST)
        options.setServerPort(PORT)
        return blpapi.Session(options, self.process_event)

    def run(self):
        session = None

        # FIX: retry loop around session start
        for attempt in range(1, MAX_SESSION_RETRIES + 1):
            session = self._build_session()
            if session.start():
                print(f"[SESSION] Connected to Bloomberg (attempt {attempt}).")
                break
            print(
                f"[SESSION] Failed to connect (attempt {attempt}/{MAX_SESSION_RETRIES}). "
                f"Retrying in {SESSION_RETRY_DELAY}s..."
            )
            time.sleep(SESSION_RETRY_DELAY)
        else:
            raise RuntimeError(
                f"Could not connect to Bloomberg after {MAX_SESSION_RETRIES} attempts."
            )

        if not session.openService("//blp/mktdata"):
            session.stop()
            raise RuntimeError("Failed to open Bloomberg market data service //blp/mktdata")

        subs = blpapi.SubscriptionList()
        subs.add(SECURITY, FIELD)
        session.subscribe(subs)

        print("=" * 50)
        print(f"  Security : {SECURITY}")
        print(f"  Field    : {FIELD}")
        print(f"  Side     : {SIDE}")
        print(f"  Distance : {STOP_DISTANCE}")
        print(f"  Entry    : {ENTRY_PRICE if ENTRY_PRICE is not None else 'first sampled tick'}")
        print(f"  Sampling : every {SAMPLE_SECONDS}s")
        print(f"  CSV      : {CSV_FILE}")
        print("=" * 50)

        try:
            while not self._stop_requested:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user.")
        finally:
            print("[SESSION] Stopping Bloomberg session...")
            session.stop()
            print("[SESSION] Done.")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    app = LiveTrailingStopApp()
    app.run()
