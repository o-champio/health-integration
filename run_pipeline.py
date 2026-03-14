"""Entry point for running the health data pipeline.

Usage:
    python run_pipeline.py                          # full date range, daily only
    python run_pipeline.py --start 2025-02-01       # custom start
    python run_pipeline.py --highfreq               # also build high-freq dataset
    python run_pipeline.py --no-incremental          # force full re-fetch
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

from src.processing.pipeline import build_daily_dataset, build_highfreq_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Health data pipeline")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--highfreq", action="store_true", help="Also build high-frequency dataset")
    parser.add_argument("--no-incremental", action="store_true", help="Force full re-fetch")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("run_pipeline")

    end = args.end or date.today().strftime("%Y-%m-%d")

    log.info("Building daily dataset (%s .. %s)...", args.start or "auto", end)
    daily = build_daily_dataset(
        start_date=args.start,
        end_date=end,
        incremental=not args.no_incremental,
    )
    log.info("Daily dataset: %d rows x %d cols", *daily.shape)

    if args.highfreq:
        start = args.start
        if start is None:
            start = daily["date"].min().strftime("%Y-%m-%d")
        log.info("Building high-frequency dataset (%s .. %s)...", start, end)
        hf = build_highfreq_dataset(start_date=start, end_date=end)
        log.info("High-freq dataset: %d rows x %d cols", *hf.shape)


if __name__ == "__main__":
    main()
