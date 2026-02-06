# Binance USD‑M Futures TESTNET Paper Bot + Dashboard

**Testnet only.** Uses:
- REST: `https://demo-fapi.binance.com`
- Market WS: `wss://fstream.binancefuture.com`

This bot is **paper trading**: it does **not** place any orders on Binance (even on testnet). It only consumes public market data and simulates maker fills locally.

## Quick start (Windows)
1. Unzip
2. Double‑click `launcher.bat`
3. Browser opens at `http://127.0.0.1:8000`
4. Press **Start**

## Quick start (Linux/Mac)
```bash
chmod +x launcher.sh
./launcher.sh
```

## Connect “your Binance testnet”
You **don’t need API keys** for this project, because it never sends orders to Binance. Your personal Binance testnet account is not used.

If you ever want *real order placement on testnet* (still not mainnet), that’s a different mode and would require adding signed REST/WebSocket calls + API keys—this project intentionally does not include that.

## Make it trade more / less (no real orders)
Defaults are tuned to be more active on testnet. You can tweak parameters via a `.env` file:

1. Copy `.env.example` to `.env`
2. Edit values
3. Restart `launcher.bat`

Useful knobs:
- `STRATEGY_ENTRY_SCORE` (lower = enters more)
- `ORDER_STALE_MS` (higher = lets orders rest longer, fewer cancels)
- `QUEUE_AHEAD_FACTOR` (lower = assume less queue ahead, more fills)
- `SYNTH_FILL_FROM_BOOK=true` (infer consumption from book size reductions when aggTrades are quiet)

## Common issues
### Browser shows ERR_CONNECTION_REFUSED
That means the server is not running or exited with an error.
Run `launcher.bat` from a terminal window to see logs.

### WS connects but there are few/no fills
On testnet, trades can be sporadic. This build includes **synthetic fills from book size reductions** (`SYNTH_FILL_FROM_BOOK=true`) to make paper fills more “live” while staying maker-only.

### Corporate proxy / VPN
If your environment blocks WebSockets, WS will fail to connect. The dashboard still loads,
but `ws_connected=false` and the risk manager will keep the bot stopped.

## Build EXE (optional)
Run `tools/build_exe.bat`. Output: `dist/PaperBotDashboard.exe`

> Note: EXE build is best-effort. For maximum reliability, use `launcher.bat`.


## Offline tooling (Phases 2.2–2.3)

### Offline backtest runner (Phase 2.2)
Replays a recorded NDJSON log and exports `trades.csv`, `orders.csv`, `equity.csv`, `summary.json`.

```bash
python -m backend.app.backtest.offline_runner --log path/to/recording.ndjson --out out/backtest_run
```

You can also replay a slice by timestamps (ms):

```bash
python -m backend.app.backtest.offline_runner --log path/to/recording.ndjson --out out/slice --t0-ms 1700000000000 --t1-ms 1700003600000
```

### Walk-forward + bounded parameter sweep (Phase 2.3)
Evaluates a bounded grid of strategy/execution params against the offline backtest metrics.

Simple sweep (uses a small built-in default grid if none is provided):

```bash
python -m backend.app.backtest.sweep_runner --log path/to/recording.ndjson --out out/sweep
```

Custom grid (CLI):

```bash
python -m backend.app.backtest.sweep_runner \
  --log path/to/recording.ndjson --out out/sweep \
  --grid SWING_MIN_EDGE_MULT=0.9,1.0,1.1,1.2 \
  --grid SWING_MAX_HOLD_MS=120000,240000,360000 \
  --grid SWING_TP_MULT_FEES=1.0,1.25 \
  --grid SWING_SL_MULT_FEES=1.0,1.25
```

Walk-forward (train->select->test):

```bash
python -m backend.app.backtest.sweep_runner \
  --walk-forward \
  --log path/to/recording.ndjson --out out/walk_forward \
  --train-ms 21600000 --test-ms 7200000 --step-ms 7200000
```

Outputs:
- `sweep_results.csv` / `sweep_results.json` (sweep)
- `top_runs/` with full CSV exports for top-N parameter sets
- `walk_forward.csv` / `walk_forward_summary.json` and per-window folders (walk-forward)

