"""Entry point for running the health data pipeline.

Usage:
    python run_pipeline.py                          # sync all (incremental)
    python run_pipeline.py --start 2025-02-01       # custom start
    python run_pipeline.py --no-incremental          # force full re-fetch
    python run_pipeline.py -v                        # debug logging
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date


def main() -> None:
    parser = argparse.ArgumentParser(description="Health data pipeline")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--no-incremental", action="store_true", help="Force full re-fetch (deletes cached parquets)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("run_pipeline")

    if args.no_incremental:
        from src.processing.pipeline import (
            GLUCOSE_PARQUET, DAILY_PARQUET, HIGHFREQ_PARQUET, WORKOUT_PARQUET,
        )
        for p in [GLUCOSE_PARQUET, DAILY_PARQUET, HIGHFREQ_PARQUET, WORKOUT_PARQUET]:
            if p.exists():
                p.unlink()
                log.info("Deleted %s", p)

    from src.processing.pipeline import sync_all, build_daily_dataset

    log.info("Running incremental sync...")
    results = sync_all()

    for name, df in results.items():
        log.info("%-12s %d rows x %d cols", name, *df.shape)


if __name__ == "__main__":
    main()
