"""Exhaustive search over feature subsets (size 1–N) to find the combination
that yields the highest adjusted R² predicting glucose_tir.

Each worker process loads the parquet once via an initializer, then evaluates
many subsets without re-reading disk.

Usage
-----
    python tests/tir_feature_search.py
    python tests/tir_feature_search.py --max-size 4 --top 20
    python tests/tir_feature_search.py --target glucose_cv --max-size 3
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Worker state (one copy per process, set by initializer) ──────────────────
_df = None
_target: str = "glucose_tir"


def _init_worker(parquet_path: str, target: str) -> None:
    """Load and build the analysis DataFrame once per worker process."""
    import warnings
    warnings.filterwarnings("ignore")

    global _df, _target
    import pandas as pd
    from src.processing.features import build_analysis_df

    _df = build_analysis_df(pd.read_parquet(parquet_path))
    _target = target


def _evaluate(combo: tuple[str, ...]) -> tuple[tuple, float, float, int] | None:
    """Evaluate one feature subset. Returns (combo, r2, r2_adj, n) or None."""
    try:
        from src.models.analysis import run_regression
        result = run_regression(_df, _target, list(combo))
        return (combo, result.r_squared, result.r_squared_adj, result.n_observations)
    except Exception:
        return None


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search for the feature subset with the highest adjusted R² for TIR"
    )
    parser.add_argument("--target", default="glucose_tir",
                        help="Glucose target column (default: glucose_tir)")
    parser.add_argument("--max-size", type=int, default=5,
                        help="Max number of features per subset (default: 5)")
    parser.add_argument("--top", type=int, default=15,
                        help="Number of top results to print (default: 15)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel worker processes (default: CPU count)")
    args = parser.parse_args()

    parquet_path = ROOT / "data" / "processed" / "daily_merged.parquet"
    if not parquet_path.exists():
        sys.exit(f"ERROR: {parquet_path} not found. Run the pipeline first.")

    # ── Discover candidate features ──────────────────────────────────────────
    import pandas as pd
    from src.processing.features import build_analysis_df, get_feature_columns

    daily = pd.read_parquet(str(parquet_path))
    df = build_analysis_df(daily)

    feat_groups = get_feature_columns(df)
    candidates: list[str] = (
        feat_groups.get("sleep_lag", [])
        + feat_groups.get("activity_lag", [])
        + feat_groups.get("derived", [])
    )
    # Keep only columns that actually exist and aren't the target
    candidates = [c for c in candidates if c in df.columns and c != args.target]

    if not candidates:
        sys.exit("No candidate feature columns found in the dataset.")

    print(f"Target         : {args.target}")
    print(f"Candidate cols : {len(candidates)}")
    for c in candidates:
        print(f"  {c}")

    # ── Build all combinations ────────────────────────────────────────────────
    combos: list[tuple[str, ...]] = []
    for size in range(1, min(args.max_size, len(candidates)) + 1):
        combos.extend(itertools.combinations(candidates, size))

    total = len(combos)
    print(f"\nMax subset size : {args.max_size}")
    print(f"Total combos    : {total:,}")
    print(f"Workers         : {args.workers or 'auto (CPU count)'}\n")

    # ── Parallel evaluation ───────────────────────────────────────────────────
    results: list[tuple] = []
    t0 = time.time()
    parquet_str = str(parquet_path)

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(parquet_str, args.target),
    ) as pool:
        futures = {pool.submit(_evaluate, combo): combo for combo in combos}
        done = 0
        for fut in as_completed(futures):
            done += 1
            if done % 200 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(
                    f"  {done:>6,}/{total:,}  "
                    f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)    ",
                    end="\r",
                )
            out = fut.result()
            if out is not None:
                results.append(out)

    elapsed_total = time.time() - t0
    print(f"\n\nDone: {len(results):,} successful regressions in {elapsed_total:.1f}s\n")

    if not results:
        print("No successful regressions — dataset may not have enough observations.")
        return

    # ── Report ────────────────────────────────────────────────────────────────
    results.sort(key=lambda x: x[2], reverse=True)  # sort by adj R²

    col_w = 70
    print(f"{'Rank':<5} {'Adj R²':>7} {'R²':>7} {'N':>5}  Features")
    print("-" * (col_w + 26))
    for rank, (combo, r2, r2_adj, n) in enumerate(results[: args.top], 1):
        feats = ", ".join(combo)
        if len(feats) > col_w:
            feats = feats[: col_w - 3] + "..."
        print(f"{rank:<5} {r2_adj:>7.4f} {r2:>7.4f} {n:>5}  {feats}")

    best_combo, best_r2, best_r2_adj, best_n = results[0]
    print(f"\n{'='*60}")
    print(f"Best subset  Adj R² = {best_r2_adj:.4f}  |  R² = {best_r2:.4f}  |  n = {best_n}")
    print(f"{'='*60}")
    for f in best_combo:
        print(f"  + {f}")

    # ── Per-size best ─────────────────────────────────────────────────────────
    print("\nBest Adj R² by subset size:")
    for size in range(1, args.max_size + 1):
        size_results = [r for r in results if len(r[0]) == size]
        if size_results:
            combo, r2, r2_adj, n = size_results[0]
            print(f"  size={size}  adj_r2={r2_adj:.4f}  [{', '.join(combo)}]")


if __name__ == "__main__":
    main()
