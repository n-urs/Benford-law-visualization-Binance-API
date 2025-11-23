# Benford-law-visualization-Binance-API

<img width="2068" height="1069" alt="image" src="https://github.com/user-attachments/assets/1c9ab343-21e1-42e1-aedb-4ca56e89fdc8" />

Interactive PyQt6 app that fetches **executed trades** from Binance (Spot and USDT perpetual futures), builds a **Benford distribution** for the **first two significant digits** (10..99), and visualizes four panels for a **main window** and a **control window** — for both **Spot** and **Futures**. It supports **aggregated trades** (`aggTrades`) and **raw trades** (`historicalTrades` / `trades`), incremental CSV writes, and a χ² (chi‑square) goodness‑of‑fit test.
---

## Features

* **Two tabs**

  * **USDT size** (price × quantity, exact with `Decimal`) — 4 panels (Spot/Futures × Main/Control)
  * **Quantity** — 4 panels (Spot/Futures × Main/Control)
* **Modes**

  * **Aggregated (aggTrades)**: high‑throughput grouped trades
  * **RAW (historicalTrades)**: individual executed trades
* **Controls**

  * Start / Stop (graceful finalize) / Continue / End (abort)
  * Custom **main** and **control** windows (start/end date & time)
  * Pair input (e.g., `CFX/USDT`)
* **Live progress** with detailed logging; **partial updates every 10k rows**
* **CSV** written **next to the script** (`benford_trades_dataset.csv`) and updated incrementally
* **Plots**

  * X axis: digits **10..99** (all values are plotted; tick labels: **10,20,30,…,90,99**)
  * Y axis: **probability 0.0–0.1**
  * **Tiny dots only** (no lines) for **Observed** and **Benford expected**
  * **χ²** stats box on each panel (p‑value if SciPy present)
* **Rate‑limit smartness**: 429/418 → **sleep 20s** and retry; no limit reduction
* **Futures RAW fallback**: if `/fapi/v1/historicalTrades` returns **400**, automatically retry via `/fapi/v1/trades?fromId=` then a minimal `historicalTrades` call

---

## Requirements

* Python **3.10+** (tested up to 3.13)
* Binance account + **API key** (needed for RAW mode; optional for agg)
* Packages:

  * `PyQt6`, `matplotlib`, `pandas`, `requests`, `python-dotenv`
  * `scipy` *(optional, enables p‑value for χ²)*

### Quick install

```bash
python -m venv .venv
# Windows
. .venv/Scripts/activate
# macOS/Linux
# source .venv/bin/activate

pip install PyQt6 matplotlib pandas requests python-dotenv
# Optional (for χ² p‑value)
pip install scipy
```

> You can also create a `requirements.txt`:
>
> ```txt
> PyQt6
> matplotlib
> pandas
> requests
> python-dotenv
> scipy    # optional
> ```

---

## Configuration

Create a `config.env` next to the script:

```env
# Required for RAW (historicalTrades) calls, recommended for all
BINANCE_API_KEY=your_api_key_here
```

The app automatically loads `config.env` from the script directory.

---

## Run

```bash
python benford_binance_visualizer_combined_en_v8.py
```

Default values are prefilled for the CFX news window (**Europe/Rome** timezone):

* Main window: `2025‑11‑11 16:55 → 17:55`
* Control window: defaults to **48 hours earlier** (editable in the UI)
* Pair: `CFX/USDT`

---

## GUI guide

**Left panel** — parameters and buttons

* **Start / Stop / Continue / End**

  * **Start**: begins fetching (no network work happens before you press this)
  * **Stop**: pauses and finalizes with collected data; charts & CSV update
  * **Continue**: resumes after Stop
  * **End**: immediate abort (no further update)
* **Mode**: `Aggregated (aggTrades)` or `RAW (historicalTrades)`

  * **RAW requires `BINANCE_API_KEY`** in `config.env`
* **Main period**: start/end date + time
* **Control period**: start/end date + time (fully editable)
* **Pair**: like `CFX/USDT`
* **Status area**: streaming logs (pages fetched, totals, sleeps on rate‑limit, etc.)

**Tabs**

1. **USDT size**

   * SPOT • Main / FUTURES • Main / SPOT • Control / FUTURES • Control
2. **Quantity (Qty)**

   * Same four panels

**Axes**

* **X**: first two significant digits in **10..99** (we look at the two leading **significant** digits; e.g., `0.0978`→`97`, `1`→`10`)

  * Tick labels: **10,20,30,…,90,99** *(the value **11** is still considered in computations but removed from tick labels for clarity)*
  * Slight x‑padding (**9.5..99.5**) so dots at **10** and **99** are readable
* **Y**: **probability**, fixed **0.0–0.1** scale
* **Markers**: dots only (very small) for *Observed* and *Benford expected*
* **χ² box**: `χ², df, p, N` (p requires SciPy)

---

## Data & files

* **CSV**: `benford_trades_dataset.csv` (script directory)

  * Columns (agg mode):

    * `ts` (ms), `price`, `qty`, `notional_usdt`, `isBuyerMaker`, `aggId`, `firstId`, `lastId`, `market` (`spot|futures`), `window` (`main|control`), `symbol`, `mode` (`agg`), `ts_iso_utc`
  * Columns (raw mode):

    * `id`, `ts` (ms), `price`, `qty`, `notional_usdt`, `isBuyerMaker`, `market`, `window`, `symbol`, `mode` (`raw`), `ts_iso_utc`
* **Logs**: `benford_log.txt` (script directory)
* **Updates**: CSV and charts refresh **every 10,000 rows** accumulated

---

## How fetching works

* **Aggregated (aggTrades)**

  * Endpoints: `/api/v3/aggTrades` (Spot) and `/fapi/v1/aggTrades` (Futures)
  * We page by time with `startTime/endTime` and stitch all pages
* **RAW (historicalTrades)**

  * Endpoints: `/api/v3/historicalTrades` and `/api/v3/trades` (Spot); `/fapi/v1/historicalTrades` and `/fapi/v1/trades` (Futures)
  * We binary‑search by `id` to find the exact `fromId` range for your windows, then fetch sequential pages
  * If **Futures** `historicalTrades` returns **400**, we **fallback** to `trades?fromId=…`, then try a minimal `historicalTrades` call
* **Rate limits**: on **429/418** we sleep **20 seconds** and retry (no limit reduction)

---

## Benford details

We use the two‑digit Benford distribution on **10..99** with probabilities

[ p(k) = \log_{10}!\left(1 + \frac{1}{k}\right) ] (normalized over 10..99),

and compare **Observed** vs **Expected** using a **χ² (chi‑square) goodness‑of‑fit** test.

> Interpretation tip: very small p‑values suggest the observed distribution deviates from the Benford law for this window; always consider **liquidity, tick size, rounding rules**, and **exchange matching mechanics** before concluding manipulation.

---

## Troubleshooting

* **No dots / flat plots**: likely zero rows in that panel (check the status log and time ranges). Ensure the main/control windows really contain trades and the pair exists on that market.
* **RAW mode error “requires BINANCE_API_KEY”**: set your key in `config.env`.
* **Futures RAW 400**: the script auto‑falls back to `/fapi/v1/trades`. See logs for “fallback”.
* **Rate‑limit sleeps**: on 429/418 the app logs a 20‑second sleep. This is expected under load.
* **Timezone**: inputs are parsed in **Europe/Rome**; all timestamps in CSV include `ts_iso_utc`.

---

## Roadmap

* Toggle **counts ↔ probabilities**
* Export charts to PNG/PDF
* Configurable **partial‑update batch size**
* Additional tests (first‑digit, three‑digit, mantissa) and side‑by‑side comparisons

---

## License

Add your preferred license (e.g., MIT).

---

## Credits

* Binance REST APIs: Spot & Futures
* Benford law references in forensic auditing research
