#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benford (two leading digits 10..99) • Binance Spot & USDT Perp Futures (agg/raw)

v8 changes:
- X-axis tick labels are now: 10,20,30,...,90,99 (11 removed from labels, still plotted).
- X-limits padded to (9.5, 99.5) so 10 and 99 dots are clearly visible.
- Tabs reordered: **USDT size** first page, **Qty** second page.
- RAW (futures) robustness: if /fapi/v1/historicalTrades returns 400, try /fapi/v1/trades with fromId,
  then a minimal variant (historicalTrades with only fromId). get_trade_time_at_id() also falls back.
- Keeps all prior functionality: control period, Start/Stop/Continue/End, partial updates every 10k,
  exact USDT notional via Decimal, 429/418 -> sleep 20s.
"""

import os
import sys
import time
import math
import logging
import traceback
from typing import Optional, Tuple, Dict, Callable, List
from datetime import datetime, timedelta, timezone
from decimal import Decimal, getcontext

import requests
import pandas as pd

# Increase precision for notional multiplication
getcontext().prec = 40

# SciPy (optional) for p-value
try:
    from scipy.stats import chi2 as _scipy_chi2
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# Timezone
try:
    from zoneinfo import ZoneInfo
    TZ_ROME = ZoneInfo("Europe/Rome")
except Exception:
    TZ_ROME = timezone(timedelta(hours=1))

# Paths (script dir) + .env load there too
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
CSV_OUT = os.path.join(SCRIPT_DIR, "benford_trades_dataset.csv")
LOG_FILE = os.path.join(SCRIPT_DIR, "benford_log.txt")
ENV_PATH = os.path.join(SCRIPT_DIR, "config.env")

# .env
try:
    from dotenv import load_dotenv
    load_dotenv(ENV_PATH)
except Exception:
    pass

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "").strip()

COMMON_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) BenfordAnalyzer/2.4",
    "Accept": "*/*",
    "Connection": "keep-alive",
}
if BINANCE_API_KEY:
    COMMON_HEADERS["X-MBX-APIKEY"] = BINANCE_API_KEY

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("benford")

# Endpoints
SPOT_AGG = "https://api.binance.com/api/v3/aggTrades"
FUT_AGG  = "https://fapi.binance.com/fapi/v1/aggTrades"
SPOT_TRADES_RECENT = "https://api.binance.com/api/v3/trades"
SPOT_TRADES_HIST   = "https://api.binance.com/api/v3/historicalTrades"
FUT_TRADES_RECENT  = "https://fapi.binance.com/fapi/v1/trades"
FUT_TRADES_HIST    = "https://fapi.binance.com/fapi/v1/historicalTrades"

# ---------- Helpers ----------

def normalize_symbol(sym: str) -> str:
    return sym.upper().replace("/", "").replace("-", "").replace(" ", "")

def to_utc_ms(dt_local: datetime) -> int:
    if dt_local.tzinfo is None:
        dt_local = dt_local.replace(tzinfo=TZ_ROME)
    return int(dt_local.astimezone(timezone.utc).timestamp() * 1000)

def parse_local_datetime(date_str: str, time_str: str) -> datetime:
    dt = datetime.strptime(f"{date_str.strip()} {time_str.strip()}", "%Y-%m-%d %H:%M")
    return dt.replace(tzinfo=TZ_ROME)

def benford_two_digit_probs_10_99() -> Dict[int, float]:
    """Benford two-digit probabilities p(k) = log10(1 + 1/k), normalized on 10..99."""
    probs = {k: math.log10(1.0 + 1.0/k) for k in range(10, 100)}
    total = sum(probs[k] for k in range(10, 100))
    return {k: probs[k] / total for k in range(10, 100)}

BENFORD_P = benford_two_digit_probs_10_99()

def expected_counts_from_total(total: int) -> Dict[int, float]:
    return {k: BENFORD_P[k] * total for k in range(10, 100)}

def chi_square(obs: Dict[int, int], exp: Dict[int, float]):
    chi = 0.0
    k = 0
    for d in range(10, 100):
        Ei = exp.get(d, 0.0)
        Oi = float(obs.get(d, 0))
        if Ei > 0:
            chi += (Oi - Ei) ** 2 / Ei
            k += 1
    df = max(k - 1, 1)
    p = _scipy_chi2.sf(chi, df) if _HAVE_SCIPY else None
    return chi, df, p

def extract_two_sig_digits(number_str: str) -> int | None:
    """
    Return first two **significant** digits as an integer in [10..99].
    Examples:
        '447.0' -> 44
        '0.0978' -> 97
        '1' -> 10
        '0.09' -> 90
    """
    raw = ''.join(ch for ch in str(number_str) if ch.isdigit())
    raw = raw.lstrip('0')
    if len(raw) >= 2:
        val = int(raw[:2])
    elif len(raw) == 1:
        val = int(raw[0] + '0')
    else:
        return None
    return val if 10 <= val <= 99 else None

def compute_distribution_from_series(series: pd.Series) -> Dict[int, int]:
    buckets = {i: 0 for i in range(10, 100)}
    if series is None or len(series) == 0:
        return buckets
    vals = series.astype(str).map(extract_two_sig_digits)
    counts = vals.value_counts(dropna=True)
    for k, v in counts.items():
        if pd.notna(k) and isinstance(k, (int,)) and 10 <= k <= 99:
            buckets[int(k)] = int(v)
    return buckets

# ---------- HTTP helpers ----------

def api_get_json(url: str, params: dict, headers=None, timeout=25):
    """GET with 429/418 policy: sleep 20s and retry; no limit reduction."""
    while True:
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code in (429, 418):
                log.warning("HTTP %s from %s — sleeping 20s, then retrying.", r.status_code, url)
                time.sleep(20)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.warning("HTTP error from %s (%s) — sleeping 10s then retrying.", url, e)
            time.sleep(10)

# ---------- RAW helpers (futures fallbacks) ----------

def _futures_hist_variants(session: requests.Session, symbol: str, from_id: int, limit: int, headers=None):
    """Try several variants to fetch one page of FUTURES raw trades around a fromId."""
    fut_base = "https://fapi.binance.com"
    urls_params = [
        (fut_base + "/fapi/v1/historicalTrades", {"symbol": symbol, "fromId": from_id, "limit": limit}),   # preferred
        (fut_base + "/fapi/v1/trades",           {"symbol": symbol, "fromId": from_id, "limit": limit}),   # alt: trades w/ fromId
        (fut_base + "/fapi/v1/historicalTrades", {"symbol": symbol, "fromId": from_id}),                   # minimal
    ]
    for url, params in urls_params:
        try:
            r = session.get(url, params=params, timeout=25)
            if r.status_code in (429, 418):
                log.warning("FUT futures page got %s on %s — sleep 20s", r.status_code, url)
                time.sleep(20)
                continue
            if r.status_code == 400:
                log.warning("FUT futures page got 400 on %s with params=%s, trying next variant.", url, params)
                continue
            r.raise_for_status()
            data = r.json()
            if data:
                return data
        except Exception as e:
            log.warning("FUT futures page exception on %s: %s — sleep 10s", url, e)
            time.sleep(10)
    # if all variants failed, raise
    raise requests.HTTPError("All futures variants failed for fromId=%s" % from_id)

# ---------- Streaming fetchers ----------

def fetch_agg_trades_stream(base_url: str, symbol: str, start_ms: int, end_ms: int, headers,
                            page_limit: int,
                            on_page: Callable[[List[dict]], None],
                            should_stop: Callable[[], bool],
                            wait_if_paused: Callable[[], None],
                            label: str):
    log.info("aggTrades %s: %s %s..%s", label, symbol, start_ms, end_ms)
    current = start_ms
    s = requests.Session()
    if headers:
        s.headers.update(headers)
    pages = 0
    total = 0
    while current <= end_ms:
        if should_stop(): break
        wait_if_paused()
        params = {"symbol": symbol, "startTime": current, "endTime": end_ms, "limit": page_limit}
        try:
            r = s.get(base_url, params=params, timeout=25)
            if r.status_code in (429, 418):
                log.warning("aggTrades %s got %s — sleep 20s", label, r.status_code)
                time.sleep(20)
                continue
            r.raise_for_status()
            data = r.json()
            if not data:
                current += 1000
                continue
            on_page(data)
            pages += 1
            total += len(data)
            last_T = int(data[-1].get("T", current))
            current = max(current + 1, last_T + 1)
            if pages % 10 == 0:
                log.info("aggTrades %s pages=%d, total=%d", label, pages, total)
        except Exception as e:
            log.warning("aggTrades %s page error: %s — sleep 10s", label, e)
            time.sleep(10)
    log.info("aggTrades %s finished. Total=%d", label, total)

def fetch_raw_trades_stream(base_hist_url: str, base_recent_url: str, symbol: str,
                            start_ms: int, end_ms: int, headers,
                            on_page: Callable[[List[dict]], None],
                            should_stop: Callable[[], bool],
                            wait_if_paused: Callable[[], None],
                            label: str):
    last_id, _ = get_latest_trade_id(base_recent_url, symbol, headers=headers)
    start_id = bsearch_first_id_at_or_after(base_hist_url, symbol, start_ms, last_id, headers=headers)
    end_id   = bsearch_first_id_after(base_hist_url, symbol, end_ms,   last_id, headers=headers)
    log.info("RAW %s id range: [%s, %s)", label, start_id, end_id)

    s = requests.Session()
    if headers:
        s.headers.update(headers)
    current = start_id
    total = 0
    is_futures = ("fapi.binance.com" in base_hist_url)

    while current < end_id:
        if should_stop(): break
        wait_if_paused()
        limit = min(1000, end_id - current)
        if is_futures:
            try:
                data = _futures_hist_variants(s, symbol, current, limit, headers=headers)
            except Exception as e:
                log.error("RAW FUTURES fetch failed at fromId=%s: %s", current, e)
                # give some time and retry next loop
                time.sleep(10)
                continue
        else:
            params = {"symbol": symbol, "fromId": current, "limit": limit}
            r = s.get(base_hist_url, params=params, timeout=25)
            if r.status_code in (429, 418):
                log.warning("RAW %s got %s — sleep 20s", label, r.status_code)
                time.sleep(20)
                continue
            if r.status_code == 400:
                log.warning("RAW %s got 400 at fromId=%s — retry in 10s", label, current)
                time.sleep(10)
                continue
            r.raise_for_status()
            data = r.json()

        if not data:
            break
        on_page(data)
        total += len(data)
        # advance by last id
        try:
            current = int(data[-1]["id"]) + 1
        except Exception:
            # some responses may use different key; try 'a' (aggId) fallback
            if isinstance(data[-1], dict) and "id" not in data[-1]:
                log.warning("RAW %s unexpected record keys: %s", label, list(data[-1].keys()))
            break

        if total % 10000 < 1000:
            log.info("RAW %s total so far: %d", label, total)
        time.sleep(0.05)
    log.info("RAW %s finished. Total=%d", label, total)

# ----- Utilities used by fetchers -----

def get_latest_trade_id(base_recent_url: str, symbol: str, headers=None) -> Tuple[int, int]:
    params = {"symbol": symbol, "limit": 1}
    data = api_get_json(base_recent_url, params, headers=headers)
    if not data:
        raise RuntimeError("Empty response from /trades")
    last = data[-1]
    return int(last["id"]), int(last["time"])

def get_trade_time_at_id(base_hist_url: str, symbol: str, trade_id: int, headers=None) -> Optional[int]:
    s = requests.Session()
    if headers:
        s.headers.update(headers)
    # primary request
    params = {"symbol": symbol, "fromId": trade_id, "limit": 1}
    try:
        r = s.get(base_hist_url, params=params, timeout=25)
        if r.status_code in (429, 418):
            log.warning("get_trade_time_at_id %s got %s — sleep 20s", base_hist_url, r.status_code)
            time.sleep(20)
            return get_trade_time_at_id(base_hist_url, symbol, trade_id, headers=headers)
        if r.status_code == 400 and "fapi.binance.com" in base_hist_url:
            # fallback to /fapi/v1/trades with fromId
            try:
                r2 = s.get(FUT_TRADES_RECENT, params={"symbol": symbol, "fromId": trade_id, "limit": 1}, timeout=25)
                if r2.status_code in (429, 418):
                    log.warning("fallback get_trade_time_at_id got %s — sleep 20s", r2.status_code)
                    time.sleep(20)
                    return get_trade_time_at_id(base_hist_url, symbol, trade_id, headers=headers)
                r2.raise_for_status()
                d2 = r2.json()
                if d2:
                    return int(d2[0].get("time", 0))
            except Exception as e2:
                log.warning("fallback get_trade_time_at_id exception: %s", e2)
                return None
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        return int(data[0].get("time", 0))
    except Exception as e:
        log.warning("get_trade_time_at_id error: %s", e)
        return None

def bsearch_first_id_at_or_after(base_hist_url: str, symbol: str, target_ms: int, max_id: int, headers=None) -> int:
    lo, hi = 0, max_id
    newest_time = get_trade_time_at_id(base_hist_url, symbol, max_id, headers=headers)
    if newest_time is None:
        raise RuntimeError("Cannot get time for max_id")
    if newest_time < target_ms:
        return max_id + 1
    while lo < hi:
        mid = (lo + hi) // 2
        t = get_trade_time_at_id(base_hist_url, symbol, mid, headers=headers)
        if t is None:
            lo = mid + 1; continue
        if t < target_ms:
            lo = mid + 1
        else:
            hi = mid
    return lo

def bsearch_first_id_after(base_hist_url: str, symbol: str, target_ms: int, max_id: int, headers=None) -> int:
    lo, hi = 0, max_id
    newest_time = get_trade_time_at_id(base_hist_url, symbol, max_id, headers=headers)
    if newest_time is None or newest_time <= target_ms:
        return max_id + 1
    while lo < hi:
        mid = (lo + hi) // 2
        t = get_trade_time_at_id(base_hist_url, symbol, mid, headers=headers)
        if t is None:
            lo = mid + 1; continue
        if t <= target_ms:
            lo = mid + 1
        else:
            hi = mid
    return lo

# ---------- DataFrame builders ----------

def _decimal_str_mul(a: str, b: str) -> str:
    try:
        res = (Decimal(a) * Decimal(b)).normalize()
        # convert to plain string without scientific notation
        s = format(res, 'f')
        # remove trailing zeros and trailing dot
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
        return s if s else "0"
    except Exception:
        return "0"

def build_df_from_agg(rows: list, symbol: str, market: str, window: str) -> pd.DataFrame:
    cols = ["ts","price","qty","notional_usdt","isBuyerMaker","aggId","firstId","lastId","market","window","symbol","mode"]
    if not rows:
        return pd.DataFrame(columns=cols)
    recs = []
    for r in rows:
        p = str(r.get("p", "0"))
        q = str(r.get("q", "0"))
        recs.append({
            "ts": int(r.get("T", 0)),
            "price": p,
            "qty": q,
            "notional_usdt": _decimal_str_mul(p, q),
            "isBuyerMaker": bool(r.get("m", False)),
            "aggId": int(r.get("a", -1)),
            "firstId": int(r.get("f", -1)),
            "lastId": int(r.get("l", -1)),
            "market": market,
            "window": window,
            "symbol": symbol,
            "mode": "agg"
        })
    return pd.DataFrame.from_records(recs, columns=cols)

def build_df_from_raw(rows: list, symbol: str, market: str, window: str) -> pd.DataFrame:
    cols = ["id","ts","price","qty","notional_usdt","isBuyerMaker","market","window","symbol","mode"]
    if not rows:
        return pd.DataFrame(columns=cols)
    recs = []
    for r in rows:
        p = str(r.get("price", "0"))
        q = str(r.get("qty", "0"))
        recs.append({
            "id": int(r.get("id", -1)),
            "ts": int(r.get("time", 0)),
            "price": p,
            "qty": q,
            "notional_usdt": _decimal_str_mul(p, q),
            "isBuyerMaker": bool(r.get("isBuyerMaker", False)),
            "market": market,
            "window": window,
            "symbol": symbol,
            "mode": "raw"
        })
    return pd.DataFrame.from_records(recs, columns=cols)

# ---------- Plot canvas ----------

class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(8, 3), dpi=100)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)

    def plot_distribution(self, counts: Dict[int,int], title: str, show_expected: bool = True):
        self.ax.clear()
        xs = list(range(10, 100))
        total = sum(int(counts.get(i, 0)) for i in xs)
        obs_freq = [(counts.get(i, 0) / total) if total > 0 else 0.0 for i in xs]
        exp_freq = [BENFORD_P[i] for i in xs]

        # Dots only (very small)
        self.ax.scatter(xs, obs_freq, s=8, label="Observed", zorder=3)
        if show_expected:
            self.ax.scatter(xs, exp_freq, s=8, label="Benford expected", zorder=2)

        # X ticks: 10,20,30,...,90,99 (no 11 label)
        xticks = list(range(10, 100, 10))
        if 99 not in xticks:
            xticks.append(99)
        self.ax.set_xlim(9.5, 99.5)  # small padding so 10/99 are visible
        self.ax.set_xticks(xticks + ([99] if 99 not in xticks else []))
        self.ax.set_xlabel("First two significant digits (10..99)")
        self.ax.set_ylabel("Probability")
        self.ax.grid(True, which="both", linestyle="--", alpha=0.3, axis="y")
        self.ax.set_ylim(0.0, 0.1)  # fixed Oy range

        # χ² annotation (use counts)
        expected_counts = expected_counts_from_total(total) if total > 0 else {k: 0.0 for k in xs}
        chi, df, p = chi_square({i: counts.get(i, 0) for i in xs}, expected_counts)
        if total > 0:
            if p is not None:
                subtitle = f"χ²={chi:.2f}, df={df}, p={p:.3g}, N={total}"
            else:
                subtitle = f"χ²={chi:.2f}, df={df}, N={total} (install SciPy for p-value)"
        else:
            subtitle = "N=0 (showing expected probabilities only)"

        self.ax.text(0.02, 0.95, subtitle, transform=self.ax.transAxes,
                     ha="left", va="top", fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.6, ec="gray"))
        self.ax.legend(loc="upper right", fontsize=8)

        self.ax.set_title(title)
        self.fig.tight_layout()
        self.draw()

# ---------- Worker with pause/stop and partial updates ----------

class FetchWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(str)
    partial = QtCore.pyqtSignal(object)   # emits partial distributions + counts
    finished_ok = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, start_dt_local: datetime, end_dt_local: datetime, symbol_in: str, mode: str,
                 ctrl_start_dt_local: datetime, ctrl_end_dt_local: datetime):
        super().__init__()
        self.start_dt_local = start_dt_local
        self.end_dt_local = end_dt_local
        self.ctrl_start_dt_local = ctrl_start_dt_local
        self.ctrl_end_dt_local = ctrl_end_dt_local
        self.symbol_in = symbol_in
        self.mode = mode  # 'agg' or 'raw'
        self._stop = False
        self._pause = False
        self._end = False

        # Collected data
        self.rows_spot_main = []
        self.rows_fut_main  = []
        self.rows_spot_ctl  = []
        self.rows_fut_ctl   = []
        self._last_emit_total = 0  # trigger every 10k

    # control flags
    def request_stop(self):
        self._stop = True

    def request_continue(self):
        self._pause = False

    def request_pause(self):
        self._pause = True

    def request_end(self):
        self._end = True
        self._stop = True  # also stop loop

    # helpers used by fetchers
    def should_stop(self) -> bool:
        return self._stop or self._end

    def wait_if_paused(self):
        while self._pause and not self._end:
            time.sleep(0.3)

    def emit_progress(self, msg: str):
        log.info(msg)
        self.progress.emit(msg)

    def _maybe_emit_partial(self):
        total = (len(self.rows_spot_main) + len(self.rows_fut_main) +
                 len(self.rows_spot_ctl)  + len(self.rows_fut_ctl))
        if total - self._last_emit_total >= 10_000:
            self._last_emit_total = total
            # Build current DataFrames and CSV
            df_spot_main = self._build_df("spot", "main", self.rows_spot_main)
            df_fut_main  = self._build_df("futures", "main", self.rows_fut_main)
            df_spot_ctl  = self._build_df("spot", "control", self.rows_spot_ctl)
            df_fut_ctl   = self._build_df("futures", "control", self.rows_fut_ctl)
            df_all = pd.concat([df_spot_main, df_fut_main, df_spot_ctl, df_fut_ctl], ignore_index=True)
            if not df_all.empty:
                df_all["ts_iso_utc"] = pd.to_datetime(df_all["ts"], unit="ms", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            df_all.to_csv(CSV_OUT, index=False, encoding="utf-8")
            # Compute distributions for qty AND notional
            obj = {
                # qty
                "dist_spot_main_qty": compute_distribution_from_series(df_spot_main.get("qty")),
                "dist_fut_main_qty":  compute_distribution_from_series(df_fut_main.get("qty")),
                "dist_spot_ctl_qty":  compute_distribution_from_series(df_spot_ctl.get("qty")),
                "dist_fut_ctl_qty":   compute_distribution_from_series(df_fut_ctl.get("qty")),
                # usdt notional
                "dist_spot_main_usd": compute_distribution_from_series(df_spot_main.get("notional_usdt")),
                "dist_fut_main_usd":  compute_distribution_from_series(df_fut_main.get("notional_usdt")),
                "dist_spot_ctl_usd":  compute_distribution_from_series(df_spot_ctl.get("notional_usdt")),
                "dist_fut_ctl_usd":   compute_distribution_from_series(df_fut_ctl.get("notional_usdt")),
                "total_rows": total,
                "csv_path": os.path.abspath(CSV_OUT)
            }
            self.partial.emit(obj)

    def _build_df(self, market: str, window: str, rows: list) -> pd.DataFrame:
        symbol = getattr(self, "symbol", self.symbol_in)
        if self.mode == "agg":
            return build_df_from_agg(rows, symbol, market, window)
        else:
            return build_df_from_raw(rows, symbol, market, window)

    def run(self):
        try:
            self.symbol = normalize_symbol(self.symbol_in)
            start_ms = to_utc_ms(self.start_dt_local)
            end_ms   = to_utc_ms(self.end_dt_local)
            cstart_ms = to_utc_ms(self.ctrl_start_dt_local)
            cend_ms   = to_utc_ms(self.ctrl_end_dt_local)

            if end_ms <= start_ms:
                raise ValueError("Main: end must be after start.")
            if cend_ms <= cstart_ms:
                raise ValueError("Control: end must be after start.")

            # define callbacks
            def on_page_spot_main(data): 
                self.rows_spot_main.extend(data); self._maybe_emit_partial()
            def on_page_fut_main(data): 
                self.rows_fut_main.extend(data); self._maybe_emit_partial()
            def on_page_spot_ctl(data):  
                self.rows_spot_ctl.extend(data); self._maybe_emit_partial()
            def on_page_fut_ctl(data):   
                self.rows_fut_ctl.extend(data); self._maybe_emit_partial()

            # streams
            if self.mode == "agg":
                self.emit_progress("Starting aggTrades stream: SPOT (main)...")
                fetch_agg_trades_stream(SPOT_AGG, self.symbol, start_ms, end_ms, COMMON_HEADERS, 1000,
                                        on_page_spot_main, self.should_stop, self.wait_if_paused, "SPOT main")
                if self._end: 
                    self.failed.emit("Ended by user"); return

                self.emit_progress("Starting aggTrades stream: FUTURES (main)...")
                fetch_agg_trades_stream(FUT_AGG, self.symbol, start_ms, end_ms, COMMON_HEADERS, 1000,
                                        on_page_fut_main, self.should_stop, self.wait_if_paused, "FUT main")
                if self._end: 
                    self.failed.emit("Ended by user"); return

                self.emit_progress("Starting aggTrades stream: SPOT (control)...")
                fetch_agg_trades_stream(SPOT_AGG, self.symbol, cstart_ms, cend_ms, COMMON_HEADERS, 1000,
                                        on_page_spot_ctl, self.should_stop, self.wait_if_paused, "SPOT control")
                if self._end: 
                    self.failed.emit("Ended by user"); return

                self.emit_progress("Starting aggTrades stream: FUTURES (control)...")
                fetch_agg_trades_stream(FUT_AGG, self.symbol, cstart_ms, cend_ms, COMMON_HEADERS, 1000,
                                        on_page_fut_ctl, self.should_stop, self.wait_if_paused, "FUT control")
                if self._end: 
                    self.failed.emit("Ended by user"); return

            elif self.mode == "raw":
                if not BINANCE_API_KEY:
                    raise RuntimeError("RAW mode requires BINANCE_API_KEY in config.env.")
                self.emit_progress("Starting RAW stream: SPOT (main)...")
                fetch_raw_trades_stream(SPOT_TRADES_HIST, SPOT_TRADES_RECENT, self.symbol, start_ms, end_ms,
                                        COMMON_HEADERS, on_page_spot_main, self.should_stop, self.wait_if_paused, "SPOT main")
                if self._end: 
                    self.failed.emit("Ended by user"); return

                self.emit_progress("Starting RAW stream: FUTURES (main)...")
                fetch_raw_trades_stream(FUT_TRADES_HIST, FUT_TRADES_RECENT, self.symbol, start_ms, end_ms,
                                        COMMON_HEADERS, on_page_fut_main, self.should_stop, self.wait_if_paused, "FUT main")
                if self._end: 
                    self.failed.emit("Ended by user"); return

                self.emit_progress("Starting RAW stream: SPOT (control)...")
                fetch_raw_trades_stream(SPOT_TRADES_HIST, SPOT_TRADES_RECENT, self.symbol, cstart_ms, cend_ms,
                                        COMMON_HEADERS, on_page_spot_ctl, self.should_stop, self.wait_if_paused, "SPOT control")
                if self._end: 
                    self.failed.emit("Ended by user"); return

                self.emit_progress("Starting RAW stream: FUTURES (control)...")
                fetch_raw_trades_stream(FUT_TRADES_HIST, FUT_TRADES_RECENT, self.symbol, cstart_ms, cend_ms,
                                        COMMON_HEADERS, on_page_fut_ctl, self.should_stop, self.wait_if_paused, "FUT control")
                if self._end: 
                    self.failed.emit("Ended by user"); return
            else:
                raise ValueError("Unknown mode.")

            # Final emit (also used when user pressed Stop = graceful stop)
            self.emit_progress("Building final CSV and charts...")
            df_spot_main = self._build_df("spot", "main", self.rows_spot_main)
            df_fut_main  = self._build_df("futures", "main", self.rows_fut_main)
            df_spot_ctl  = self._build_df("spot", "control", self.rows_spot_ctl)
            df_fut_ctl   = self._build_df("futures", "control", self.rows_fut_ctl)
            df_all = pd.concat([df_spot_main, df_fut_main, df_spot_ctl, df_fut_ctl], ignore_index=True)
            if not df_all.empty:
                df_all["ts_iso_utc"] = pd.to_datetime(df_all["ts"], unit="ms", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            df_all.to_csv(CSV_OUT, index=False, encoding="utf-8")

            result = {
                "df_all": df_all,
                # qty dists
                "dist_spot_main_qty": compute_distribution_from_series(df_spot_main.get("qty")),
                "dist_fut_main_qty":  compute_distribution_from_series(df_fut_main.get("qty")),
                "dist_spot_ctl_qty":  compute_distribution_from_series(df_spot_ctl.get("qty")),
                "dist_fut_ctl_qty":   compute_distribution_from_series(df_fut_ctl.get("qty")),
                # notional dists
                "dist_spot_main_usd": compute_distribution_from_series(df_spot_main.get("notional_usdt")),
                "dist_fut_main_usd":  compute_distribution_from_series(df_fut_main.get("notional_usdt")),
                "dist_spot_ctl_usd":  compute_distribution_from_series(df_spot_ctl.get("notional_usdt")),
                "dist_fut_ctl_usd":   compute_distribution_from_series(df_fut_ctl.get("notional_usdt")),
                "symbol": self.symbol,
                "csv_path": os.path.abspath(CSV_OUT),
                "log_path": os.path.abspath(LOG_FILE),
            }
            self.finished_ok.emit(result)
        except Exception as e:
            log.error("Worker error: %s", e)
            log.error("Traceback:\n%s", traceback.format_exc())
            self.failed.emit(str(e))

# ---------- Main Window with tabs (USDT first, Qty second) ----------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Benford (two digits 10..99) • Binance Spot & Futures (agg/raw)")
        self.resize(1600, 920)

        # Defaults
        default_symbol = "CFX/USDT"
        default_start_date = "2025-11-11"
        default_start_time = "16:55"
        default_end_date   = "2025-11-11"
        default_end_time   = "17:55"

        # Default control = −48h
        def_dt_start = parse_local_datetime(default_start_date, default_start_time)
        def_dt_end   = parse_local_datetime(default_end_date, default_end_time)
        ctrl_dt_start = (def_dt_start - timedelta(hours=48))
        ctrl_dt_end   = (def_dt_end   - timedelta(hours=48))

        # Main inputs
        self.input_start_date = QtWidgets.QLineEdit(default_start_date)
        self.input_start_time = QtWidgets.QLineEdit(default_start_time)
        self.input_end_date   = QtWidgets.QLineEdit(default_end_date)
        self.input_end_time   = QtWidgets.QLineEdit(default_end_time)
        self.input_symbol     = QtWidgets.QLineEdit(default_symbol)

        # Control inputs
        self.ctrl_start_date = QtWidgets.QLineEdit(ctrl_dt_start.strftime("%Y-%m-%d"))
        self.ctrl_start_time = QtWidgets.QLineEdit(ctrl_dt_start.strftime("%H:%M"))
        self.ctrl_end_date   = QtWidgets.QLineEdit(ctrl_dt_end.strftime("%Y-%m-%d"))
        self.ctrl_end_time   = QtWidgets.QLineEdit(ctrl_dt_end.strftime("%H:%M"))

        # Mode
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("Aggregated (aggTrades)", userData="agg")
        self.mode_combo.addItem("RAW (historicalTrades)", userData="raw")

        # Buttons
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_continue = QtWidgets.QPushButton("Continue")
        self.btn_end = QtWidgets.QPushButton("End")
        self._set_buttons_enabled(start=True, stop=False, cont=False, end=False)

        # Status
        self.status_lbl = QtWidgets.QLabel("Idle.")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setMinimumWidth(420)

        # Plots (two pages/tabs) — USDT first
        self.tabs = QtWidgets.QTabWidget()

        # Tab 1: USDT notional
        self.canvas_usd_1 = MplCanvas()
        self.canvas_usd_2 = MplCanvas()
        self.canvas_usd_3 = MplCanvas()
        self.canvas_usd_4 = MplCanvas()
        usd_layout = QtWidgets.QGridLayout()
        usd_layout.addWidget(self.canvas_usd_1, 0, 0)
        usd_layout.addWidget(self.canvas_usd_2, 0, 1)
        usd_layout.addWidget(self.canvas_usd_3, 1, 0)
        usd_layout.addWidget(self.canvas_usd_4, 1, 1)
        usd_box = QtWidgets.QGroupBox("First two significant digits of USDT SIZE (10..99)")
        usd_box.setLayout(usd_layout)
        usd_tab = QtWidgets.QWidget()
        ul = QtWidgets.QVBoxLayout(usd_tab)
        ul.addWidget(usd_box)
        self.tabs.addTab(usd_tab, "USDT size")

        # Tab 2: Qty
        self.canvas_qty_1 = MplCanvas()
        self.canvas_qty_2 = MplCanvas()
        self.canvas_qty_3 = MplCanvas()
        self.canvas_qty_4 = MplCanvas()
        qty_layout = QtWidgets.QGridLayout()
        qty_layout.addWidget(self.canvas_qty_1, 0, 0)
        qty_layout.addWidget(self.canvas_qty_2, 0, 1)
        qty_layout.addWidget(self.canvas_qty_3, 1, 0)
        qty_layout.addWidget(self.canvas_qty_4, 1, 1)
        qty_box = QtWidgets.QGroupBox("First two significant digits of QUANTITY (10..99)")
        qty_box.setLayout(qty_layout)
        qty_tab = QtWidgets.QWidget()
        ql = QtWidgets.QVBoxLayout(qty_tab)
        ql.addWidget(qty_box)
        self.tabs.addTab(qty_tab, "Qty")

        # Left controls
        controls_layout = QtWidgets.QGridLayout()
        r = 0
        controls_layout.addWidget(QtWidgets.QLabel("Start date (YYYY-MM-DD):"), r, 0); controls_layout.addWidget(self.input_start_date, r, 1); r+=1
        controls_layout.addWidget(QtWidgets.QLabel("Start time (HH:MM):"), r, 0); controls_layout.addWidget(self.input_start_time, r, 1); r+=1
        controls_layout.addWidget(QtWidgets.QLabel("End date (YYYY-MM-DD):"), r, 0); controls_layout.addWidget(self.input_end_date, r, 1); r+=1
        controls_layout.addWidget(QtWidgets.QLabel("End time (HH:MM):"), r, 0); controls_layout.addWidget(self.input_end_time, r, 1); r+=1
        controls_layout.addWidget(QtWidgets.QLabel("Pair (e.g., CFX/USDT):"), r, 0); controls_layout.addWidget(self.input_symbol, r, 1); r+=1

        controls_layout.addWidget(QtWidgets.QLabel("Control start date:"), r, 0); controls_layout.addWidget(self.ctrl_start_date, r, 1); r+=1
        controls_layout.addWidget(QtWidgets.QLabel("Control start time:"), r, 0); controls_layout.addWidget(self.ctrl_start_time, r, 1); r+=1
        controls_layout.addWidget(QtWidgets.QLabel("Control end date:"), r, 0); controls_layout.addWidget(self.ctrl_end_date, r, 1); r+=1
        controls_layout.addWidget(QtWidgets.QLabel("Control end time:"), r, 0); controls_layout.addWidget(self.ctrl_end_time, r, 1); r+=1

        controls_layout.addWidget(QtWidgets.QLabel("Mode:"), r, 0); controls_layout.addWidget(self.mode_combo, r, 1); r+=1

        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.addWidget(self.btn_start)
        buttons_layout.addWidget(self.btn_stop)
        buttons_layout.addWidget(self.btn_continue)
        buttons_layout.addWidget(self.btn_end)
        controls_layout.addLayout(buttons_layout, r, 0, 1, 2); r+=1

        controls_layout.addWidget(self.status_lbl, r, 0, 1, 2); r+=1

        controls_box = QtWidgets.QGroupBox("Request parameters (Europe/Rome)")
        controls_box.setLayout(controls_layout)

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addWidget(controls_box, 1)
        main_layout.addWidget(self.tabs, 3)

        central = QtWidgets.QWidget()
        central.setLayout(main_layout)
        self.setCentralWidget(central)

        # Connect buttons
        self.btn_start.clicked.connect(self.on_start_clicked)
        self.btn_stop.clicked.connect(self.on_stop_clicked)
        self.btn_continue.clicked.connect(self.on_continue_clicked)
        self.btn_end.clicked.connect(self.on_end_clicked)

        self.worker = None  # created on Start

    def _set_buttons_enabled(self, start: bool, stop: bool, cont: bool, end: bool):
        self.btn_start.setEnabled(start)
        self.btn_stop.setEnabled(stop)
        self.btn_continue.setEnabled(cont)
        self.btn_end.setEnabled(end)

    def on_start_clicked(self):
        try:
            start_dt = parse_local_datetime(self.input_start_date.text(), self.input_start_time.text())
            end_dt   = parse_local_datetime(self.input_end_date.text(),   self.input_end_time.text())
            cstart   = parse_local_datetime(self.ctrl_start_date.text(),  self.ctrl_start_time.text())
            cend     = parse_local_datetime(self.ctrl_end_date.text(),    self.ctrl_end_time.text())
            symbol   = self.input_symbol.text().strip()
            mode     = self.mode_combo.currentData()
        except Exception as e:
            self.status_lbl.setText(f"Datetime parse error: {e}")
            return

        if mode == "raw" and not BINANCE_API_KEY:
            self.status_lbl.setText("RAW mode requires BINANCE_API_KEY in config.env.")
            return

        self.status_lbl.setText(f"Starting fetch ({'RAW' if mode=='raw' else 'AGG'})...")
        self.worker = FetchWorker(start_dt, end_dt, symbol, mode, cstart, cend)
        self.worker.progress.connect(self.on_progress, Qt.ConnectionType.QueuedConnection)
        self.worker.partial.connect(self.on_partial, Qt.ConnectionType.QueuedConnection)
        self.worker.finished_ok.connect(self.on_fetch_success, Qt.ConnectionType.QueuedConnection)
        self.worker.failed.connect(self.on_fetch_failed, Qt.ConnectionType.QueuedConnection)
        self._set_buttons_enabled(start=False, stop=True, cont=False, end=True)
        self.worker.start()

    def on_stop_clicked(self):
        if self.worker:
            self.status_lbl.setText("Stopping gracefully (will finalize with partial data)...")
            self.worker.request_pause()
            self.worker.request_stop()

    def on_continue_clicked(self):
        if self.worker:
            self.worker.request_continue()
            self.status_lbl.setText("Resumed.")
            self._set_buttons_enabled(start=False, stop=True, cont=False, end=True)

    def on_end_clicked(self):
        if self.worker:
            self.status_lbl.setText("Ending now (no update).")
            self.worker.request_end()

    def on_progress(self, msg: str):
        prev = self.status_lbl.text()
        self.status_lbl.setText((prev + "\n" if prev else "") + msg)

    def on_partial(self, obj):
        # USDT charts
        self.canvas_usd_1.plot_distribution(obj["dist_spot_main_usd"], "SPOT • Main (USDT, partial)")
        self.canvas_usd_2.plot_distribution(obj["dist_fut_main_usd"],  "FUTURES • Main (USDT, partial)")
        self.canvas_usd_3.plot_distribution(obj["dist_spot_ctl_usd"],  "SPOT • Control (USDT, partial)")
        self.canvas_usd_4.plot_distribution(obj["dist_fut_ctl_usd"],   "FUTURES • Control (USDT, partial)")
        # Qty charts
        self.canvas_qty_1.plot_distribution(obj["dist_spot_main_qty"], "SPOT • Main (qty, partial)")
        self.canvas_qty_2.plot_distribution(obj["dist_fut_main_qty"],  "FUTURES • Main (qty, partial)")
        self.canvas_qty_3.plot_distribution(obj["dist_spot_ctl_qty"],  "SPOT • Control (qty, partial)")
        self.canvas_qty_4.plot_distribution(obj["dist_fut_ctl_qty"],   "FUTURES • Control (qty, partial)")
        self.status_lbl.setText(f"Partial update: total rows ~ {obj['total_rows']}. CSV: {obj['csv_path']}")

    def on_fetch_success(self, result_obj):
        try:
            # USDT final
            self.canvas_usd_1.plot_distribution(result_obj["dist_spot_main_usd"], "SPOT • Main window (USDT)")
            self.canvas_usd_2.plot_distribution(result_obj["dist_fut_main_usd"],  "FUTURES • Main window (USDT)")
            self.canvas_usd_3.plot_distribution(result_obj["dist_spot_ctl_usd"],  "SPOT • Control window (USDT)")
            self.canvas_usd_4.plot_distribution(result_obj["dist_fut_ctl_usd"],   "FUTURES • Control window (USDT)")
            # Qty final
            self.canvas_qty_1.plot_distribution(result_obj["dist_spot_main_qty"], "SPOT • Main window (qty)")
            self.canvas_qty_2.plot_distribution(result_obj["dist_fut_main_qty"],  "FUTURES • Main window (qty)")
            self.canvas_qty_3.plot_distribution(result_obj["dist_spot_ctl_qty"],  "SPOT • Control window (qty)")
            self.canvas_qty_4.plot_distribution(result_obj["dist_fut_ctl_qty"],   "FUTURES • Control window (qty)")

            self.status_lbl.setText(f"Done. CSV: {result_obj['csv_path']}\nLogs: {result_obj['log_path']}")
        except Exception as e:
            self.status_lbl.setText(f"Plot update error: {e}")
        finally:
            self._set_buttons_enabled(start=True, stop=False, cont=False, end=False)
            self.worker = None

    def on_fetch_failed(self, err_msg: str):
        self.status_lbl.setText(f"Fetch error: {err_msg}. See logs: {os.path.abspath(LOG_FILE)}")
        self._set_buttons_enabled(start=True, stop=False, cont=False, end=False)
        self.worker = None

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
