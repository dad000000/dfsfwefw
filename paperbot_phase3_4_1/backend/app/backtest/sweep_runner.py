from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import random
import time
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .offline_runner import _i, _iter_ndjson, run_backtest

MS_DAY = 24 * 60 * 60 * 1000


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _coerce_scalar(s: str) -> Any:
    v = s.strip()
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    # int?
    try:
        if "." not in v and "e" not in v.lower():
            return int(v)
    except Exception:
        pass
    # float?
    try:
        return float(v)
    except Exception:
        return v


def _parse_values(expr: str) -> List[Any]:
    """
    Supports:
      - "a,b,c" lists
      - "start:stop:step" ranges (inclusive stop, numeric)
    """
    e = expr.strip()
    if ":" in e and "," not in e:
        parts = [p.strip() for p in e.split(":")]
        if len(parts) == 3:
            a = float(parts[0])
            b = float(parts[1])
            step = float(parts[2])
            if step == 0:
                return [a]
            out: List[Any] = []
            n = int(math.floor((b - a) / step)) if step != 0 else 0
            # Inclusive end; guard against float noise
            x = a
            for _ in range(max(0, n) + 2):
                if (step > 0 and x > b + 1e-12) or (step < 0 and x < b - 1e-12):
                    break
                out.append(x)
                x = x + step
            # Try to keep ints as ints where possible
            cleaned: List[Any] = []
            for z in out:
                if abs(z - round(z)) < 1e-9:
                    cleaned.append(int(round(z)))
                else:
                    cleaned.append(float(z))
            return cleaned
    return [_coerce_scalar(x) for x in e.split(",") if x.strip()]


def _parse_grid(args: argparse.Namespace) -> Dict[str, List[Any]]:
    grid: Dict[str, List[Any]] = {}
    if args.grid_json:
        with open(args.grid_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("grid_json must be a JSON object of KEY -> [values...]")
        for k, v in data.items():
            if not isinstance(k, str) or not isinstance(v, list):
                continue
            grid[k] = v

    for item in args.grid or []:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        k = k.strip()
        if not k:
            continue
        grid[k] = _parse_values(v)

    if not grid and args.use_default_grid:
        grid = {
            "SWING_MIN_EDGE_MULT": [0.9, 1.0, 1.1, 1.2],
            "SWING_MAX_HOLD_MS": [120000, 240000, 360000],
            "SWING_TP_MULT_FEES": [1.0, 1.25],
            "SWING_SL_MULT_FEES": [1.0, 1.25],
        }
    return grid


def _grid_product(grid: Dict[str, List[Any]]) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    keys = sorted(grid.keys())
    vals = [grid[k] for k in keys]
    combos = list(product(*vals)) if keys else [tuple()]
    return keys, combos


def _sample_combos(keys: List[str], combos: List[Tuple[Any, ...]], sample_n: int, seed: int) -> List[Tuple[Any, ...]]:
    if sample_n <= 0 or sample_n >= len(combos):
        return combos
    rng = random.Random(seed)
    idxs = rng.sample(range(len(combos)), k=sample_n)
    return [combos[i] for i in idxs]


def _hash_params(d: Dict[str, Any]) -> str:
    s = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def _rates_from_summary(summary: Dict[str, Any]) -> Dict[str, float]:
    dur_ms = max(1, int(summary.get("duration_ms") or 0))
    trades = int(summary.get("trades") or 0)
    cancels = int(summary.get("cancels_total") or 0)
    rejects = int(summary.get("rejects_total") or 0)
    trades_per_day = trades / (dur_ms / MS_DAY)
    cancels_per_min = cancels / (dur_ms / 60000.0)
    rejects_per_min = rejects / (dur_ms / 60000.0)
    return {
        "trades_per_day": float(trades_per_day),
        "cancels_per_min": float(cancels_per_min),
        "rejects_per_min": float(rejects_per_min),
    }


def _is_valid(summary: Dict[str, Any], *, min_pf: float, min_trades_per_day: float, max_cancels_per_min: float, max_rejects_per_min: float) -> bool:
    dur_ms = int(summary.get("duration_ms") or 0)
    if dur_ms <= 0:
        return False
    pf = float(summary.get("profit_factor_net") or 0.0)
    if pf < min_pf:
        return False
    rates = _rates_from_summary(summary)
    if rates["trades_per_day"] < min_trades_per_day:
        return False
    if rates["cancels_per_min"] > max_cancels_per_min:
        return False
    if rates["rejects_per_min"] > max_rejects_per_min:
        return False
    return True


def _score(summary: Dict[str, Any], objective: str) -> float:
    obj = objective.lower()
    net = float(summary.get("net_after_fees_usdt") or 0.0)
    pf = float(summary.get("profit_factor_net") or 0.0)
    max_dd = float(summary.get("max_dd_usdt") or 0.0)
    if obj in ("net", "net_after_fees", "net_after_fees_usdt"):
        return net
    if obj in ("pf", "profit_factor", "profit_factor_net"):
        return pf
    if obj in ("net_dd", "net_over_dd"):
        return net / (1e-9 + max_dd)
    if obj in ("net_dd_penalized", "net_penalized"):
        return net - 0.25 * max_dd
    # default
    return net


def _scan_time_bounds(log_path: str) -> Tuple[int, int, int]:
    first: Optional[int] = None
    last: Optional[int] = None
    frames = 0
    for ev in _iter_ndjson(log_path):
        if ev.get("type") != "frame":
            continue
        ts = _i(ev.get("ts_ms"))
        if ts <= 0:
            continue
        if first is None:
            first = ts
        last = ts
        frames += 1
    return int(first or 0), int(last or 0), int(frames)


async def _run_sweep_once(
    *,
    log_path: str,
    out_dir: str,
    params: Dict[str, Any],
    objective: str,
    t0_ms: Optional[int],
    t1_ms: Optional[int],
    constraints: Dict[str, float],
) -> Dict[str, Any]:
    summary = await run_backtest(
        log_path,
        out_dir,
        settings_overrides=params,
        write_outputs=False,
        t0_ms=t0_ms,
        t1_ms=t1_ms,
        progress_every_s=0.0,  # no inner progress in sweeps
    )
    score = float("-inf")
    valid = _is_valid(
        summary,
        min_pf=float(constraints["min_pf"]),
        min_trades_per_day=float(constraints["min_trades_per_day"]),
        max_cancels_per_min=float(constraints["max_cancels_per_min"]),
        max_rejects_per_min=float(constraints["max_rejects_per_min"]),
    )
    if valid:
        score = _score(summary, objective)
    rates = _rates_from_summary(summary) if int(summary.get("duration_ms") or 0) > 0 else {"trades_per_day": 0.0, "cancels_per_min": 0.0, "rejects_per_min": 0.0}
    return {
        "params": params,
        "params_hash": _hash_params(params),
        "objective": objective,
        "score": float(score),
        "valid": bool(valid),
        **summary,
        **rates,
    }


def _write_results_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "rank",
        "score",
        "valid",
        "net_after_fees_usdt",
        "profit_factor_net",
        "max_dd_usdt",
        "trades",
        "trades_per_day",
        "cancels_total",
        "cancels_per_min",
        "rejects_total",
        "rejects_per_min",
        "duration_ms",
        "params_hash",
        "params_json",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows, 1):
            w.writerow(
                {
                    "rank": i,
                    "score": float(r.get("score") or 0.0),
                    "valid": bool(r.get("valid")),
                    "net_after_fees_usdt": float(r.get("net_after_fees_usdt") or 0.0),
                    "profit_factor_net": float(r.get("profit_factor_net") or 0.0),
                    "max_dd_usdt": float(r.get("max_dd_usdt") or 0.0),
                    "trades": int(r.get("trades") or 0),
                    "trades_per_day": float(r.get("trades_per_day") or 0.0),
                    "cancels_total": int(r.get("cancels_total") or 0),
                    "cancels_per_min": float(r.get("cancels_per_min") or 0.0),
                    "rejects_total": int(r.get("rejects_total") or 0),
                    "rejects_per_min": float(r.get("rejects_per_min") or 0.0),
                    "duration_ms": int(r.get("duration_ms") or 0),
                    "params_hash": str(r.get("params_hash") or ""),
                    "params_json": json.dumps(r.get("params") or {}, sort_keys=True),
                }
            )


async def _sweep(
    *,
    log_path: str,
    out_dir: str,
    grid: Dict[str, List[Any]],
    objective: str,
    max_combos: int,
    sample_n: Optional[int],
    seed: int,
    t0_ms: Optional[int],
    t1_ms: Optional[int],
    rerun_top_n: int,
    constraints: Dict[str, float],
    progress_every_s: float,
) -> List[Dict[str, Any]]:
    keys, combos = _grid_product(grid)
    total = len(combos)
    if sample_n is not None:
        combos = _sample_combos(keys, combos, sample_n, seed)
    elif max_combos > 0 and total > max_combos:
        # deterministic truncation (bounded)
        combos = combos[:max_combos]

    _ensure_dir(out_dir)
    scratch_dir = os.path.join(out_dir, "_scratch")
    # Do not create scratch dir; run_backtest(write_outputs=False) won't write.

    rows: List[Dict[str, Any]] = []
    t_start = time.monotonic()
    last_log = t_start
    best_score = float("-inf")

    for idx, tup in enumerate(combos, 1):
        params = {k: tup[i] for i, k in enumerate(keys)}
        r = await _run_sweep_once(
            log_path=log_path,
            out_dir=scratch_dir,
            params=params,
            objective=objective,
            t0_ms=t0_ms,
            t1_ms=t1_ms,
            constraints=constraints,
        )
        rows.append(r)
        if r["valid"] and float(r["score"]) > best_score:
            best_score = float(r["score"])

        now = time.monotonic()
        if progress_every_s > 0 and now - last_log >= progress_every_s:
            print(
                f"[sweep] {idx}/{len(combos)} best_score={best_score:.4f}",
                flush=True,
            )
            last_log = now

    rows.sort(key=lambda x: (bool(x.get("valid")), float(x.get("score") or float("-inf"))), reverse=True)

    _write_results_csv(os.path.join(out_dir, "sweep_results.csv"), rows)
    with open(os.path.join(out_dir, "sweep_results.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "objective": objective,
                "grid": grid,
                "count": len(rows),
                "t0_ms": t0_ms,
                "t1_ms": t1_ms,
                "constraints": constraints,
                "best": {k: rows[0].get(k) for k in ("params", "params_hash", "score", "valid", "net_after_fees_usdt", "profit_factor_net", "max_dd_usdt", "trades", "trades_per_day", "cancels_per_min", "rejects_per_min")} if rows else None,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    # Re-run top-N with full outputs
    top_dir = os.path.join(out_dir, "top_runs")
    _ensure_dir(top_dir)
    for rank, r in enumerate(rows[: max(0, rerun_top_n)], 1):
        if not r.get("valid"):
            continue
        p = r.get("params") or {}
        h = str(r.get("params_hash") or "")
        run_dir = os.path.join(top_dir, f"top_{rank:02d}_{h}")
        _ensure_dir(run_dir)
        with open(os.path.join(run_dir, "params.json"), "w", encoding="utf-8") as f:
            json.dump(p, f, indent=2, sort_keys=True)
        _ = await run_backtest(
            log_path,
            run_dir,
            settings_overrides=p,
            write_outputs=True,
            t0_ms=t0_ms,
            t1_ms=t1_ms,
            progress_every_s=3.0,
        )

    return rows


async def _walk_forward(
    *,
    log_path: str,
    out_dir: str,
    grid: Dict[str, List[Any]],
    objective: str,
    train_ms: int,
    test_ms: int,
    step_ms: int,
    max_combos: int,
    sample_n: Optional[int],
    seed: int,
    rerun_top_n_train: int,
    constraints: Dict[str, float],
    progress_every_s: float,
) -> None:
    first_ts, last_ts, frames = _scan_time_bounds(log_path)
    if first_ts <= 0 or last_ts <= 0 or frames <= 0:
        raise RuntimeError("No frames found in log; cannot run walk-forward.")

    _ensure_dir(out_dir)
    wf_rows: List[Dict[str, Any]] = []

    t = first_ts
    win = 0
    while t + train_ms + test_ms <= last_ts:
        train0 = t
        train1 = t + train_ms
        test0 = train1
        test1 = test0 + test_ms

        wdir = os.path.join(out_dir, f"window_{win:03d}")
        _ensure_dir(wdir)

        print(
            f"[walk_forward] window={win} train=[{train0},{train1}) test=[{test0},{test1})",
            flush=True,
        )

        train_dir = os.path.join(wdir, "train_sweep")
        train_rows = await _sweep(
            log_path=log_path,
            out_dir=train_dir,
            grid=grid,
            objective=objective,
            max_combos=max_combos,
            sample_n=sample_n,
            seed=seed + win,
            t0_ms=train0,
            t1_ms=train1,
            rerun_top_n=rerun_top_n_train,
            constraints=constraints,
            progress_every_s=progress_every_s,
        )

        best = next((r for r in train_rows if r.get("valid")), train_rows[0] if train_rows else None)
        best_params = (best.get("params") if best else {}) or {}
        with open(os.path.join(wdir, "selected_params.json"), "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2, sort_keys=True)

        # Evaluate on test slice and always write outputs
        test_dir = os.path.join(wdir, "test_eval")
        _ensure_dir(test_dir)
        test_summary = await run_backtest(
            log_path,
            test_dir,
            settings_overrides=best_params,
            write_outputs=True,
            t0_ms=test0,
            t1_ms=test1,
            progress_every_s=3.0,
        )

        row = {
            "window": win,
            "train_t0_ms": train0,
            "train_t1_ms": train1,
            "test_t0_ms": test0,
            "test_t1_ms": test1,
            "params_hash": _hash_params(best_params),
            "params_json": json.dumps(best_params, sort_keys=True),
            "train_net_after_fees_usdt": float(best.get("net_after_fees_usdt") or 0.0) if best else 0.0,
            "train_profit_factor_net": float(best.get("profit_factor_net") or 0.0) if best else 0.0,
            "train_max_dd_usdt": float(best.get("max_dd_usdt") or 0.0) if best else 0.0,
            "train_trades": int(best.get("trades") or 0) if best else 0,
            "test_net_after_fees_usdt": float(test_summary.get("net_after_fees_usdt") or 0.0),
            "test_profit_factor_net": float(test_summary.get("profit_factor_net") or 0.0),
            "test_max_dd_usdt": float(test_summary.get("max_dd_usdt") or 0.0),
            "test_trades": int(test_summary.get("trades") or 0),
            "test_trades_per_day": float(_rates_from_summary(test_summary)["trades_per_day"]),
            "test_cancels_per_min": float(_rates_from_summary(test_summary)["cancels_per_min"]),
            "test_rejects_per_min": float(_rates_from_summary(test_summary)["rejects_per_min"]),
        }
        wf_rows.append(row)

        win += 1
        t += max(1, int(step_ms))

    # Write walk-forward summary
    wf_csv = os.path.join(out_dir, "walk_forward.csv")
    fields = list(wf_rows[0].keys()) if wf_rows else []
    with open(wf_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in wf_rows:
            w.writerow(r)

    total_test_net = sum(float(r["test_net_after_fees_usdt"]) for r in wf_rows)
    total_test_trades = sum(int(r["test_trades"]) for r in wf_rows)
    avg_pf = sum(float(r["test_profit_factor_net"]) for r in wf_rows) / max(1, len(wf_rows))

    with open(os.path.join(out_dir, "walk_forward_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "windows": len(wf_rows),
                "train_ms": int(train_ms),
                "test_ms": int(test_ms),
                "step_ms": int(step_ms),
                "objective": objective,
                "constraints": constraints,
                "total_test_net_after_fees_usdt": float(total_test_net),
                "total_test_trades": int(total_test_trades),
                "avg_test_profit_factor_net": float(avg_pf),
            },
            f,
            indent=2,
            sort_keys=True,
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2.3 walk-forward + bounded parameter sweep (offline)")
    p.add_argument("--log", required=True, help="Path to NDJSON recording (frames)")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--grid-json", default=None, help="JSON file with KEY -> [values...] grid")
    p.add_argument("--grid", action="append", default=[], help="Override/add grid: KEY=val1,val2 or KEY=start:stop:step")
    p.add_argument("--use-default-grid", action=argparse.BooleanOptionalAction, default=True, help="Use built-in small default grid if none provided")

    p.add_argument("--objective", default="net_after_fees", help="Objective: net_after_fees | pf | net_over_dd | net_dd_penalized")
    p.add_argument("--max-combos", type=int, default=200, help="Max combos to evaluate (bounded). If grid bigger, truncates.")
    p.add_argument("--sample", type=int, default=None, help="Randomly sample N combos (instead of truncation)")
    p.add_argument("--seed", type=int, default=1, help="Seed for sampling")

    p.add_argument("--rerun-top-n", type=int, default=3, help="Re-run top N combos with full CSV outputs")

    # constraints (kept lenient by default)
    p.add_argument("--min-pf", type=float, default=0.0, help="Minimum profit_factor_net to accept")
    p.add_argument("--min-trades-per-day", type=float, default=0.0, help="Minimum trades/day to accept")
    p.add_argument("--max-cancels-per-min", type=float, default=5.0, help="Maximum cancels/min to accept")
    p.add_argument("--max-rejects-per-min", type=float, default=1.0, help="Maximum rejects/min to accept")

    p.add_argument("--progress-every-s", type=float, default=3.0, help="Console progress interval (seconds)")

    # walk-forward
    p.add_argument("--walk-forward", action=argparse.BooleanOptionalAction, default=False, help="Enable walk-forward mode")
    p.add_argument("--train-ms", type=int, default=6 * 60 * 60 * 1000, help="Train window length (ms)")
    p.add_argument("--test-ms", type=int, default=2 * 60 * 60 * 1000, help="Test window length (ms)")
    p.add_argument("--step-ms", type=int, default=2 * 60 * 60 * 1000, help="Window step (ms)")

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    grid = _parse_grid(args)
    constraints = {
        "min_pf": float(args.min_pf),
        "min_trades_per_day": float(args.min_trades_per_day),
        "max_cancels_per_min": float(args.max_cancels_per_min),
        "max_rejects_per_min": float(args.max_rejects_per_min),
    }

    if args.walk_forward:
        asyncio.run(
            _walk_forward(
                log_path=args.log,
                out_dir=args.out,
                grid=grid,
                objective=args.objective,
                train_ms=int(args.train_ms),
                test_ms=int(args.test_ms),
                step_ms=int(args.step_ms),
                max_combos=int(args.max_combos),
                sample_n=int(args.sample) if args.sample is not None else None,
                seed=int(args.seed),
                rerun_top_n_train=max(0, int(args.rerun_top_n)),
                constraints=constraints,
                progress_every_s=float(args.progress_every_s),
            )
        )
    else:
        asyncio.run(
            _sweep(
                log_path=args.log,
                out_dir=args.out,
                grid=grid,
                objective=args.objective,
                max_combos=int(args.max_combos),
                sample_n=int(args.sample) if args.sample is not None else None,
                seed=int(args.seed),
                t0_ms=None,
                t1_ms=None,
                rerun_top_n=max(0, int(args.rerun_top_n)),
                constraints=constraints,
                progress_every_s=float(args.progress_every_s),
            )
        )


if __name__ == "__main__":
    main()

