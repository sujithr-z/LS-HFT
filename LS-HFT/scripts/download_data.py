# =============================================================================
# scripts/download_data.py
#
# Orchestration only — contains NO mathematics.
#
# Pipeline:
#   load config  →  call downloader  →  save raw data
# =============================================================================

import sys
from pathlib import Path

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import TICKER, PERIOD, INTERVAL, RAW_DATA_PATH
from src.data.downloader import download


def main() -> None:
    print(f"\n{'='*55}")
    print(f"  LS-HFT V0 — Download Raw Data")
    print(f"{'='*55}")
    print(f"  Ticker   : {TICKER}")
    print(f"  Period   : {PERIOD}")
    print(f"  Interval : {INTERVAL}")
    print(f"  Output   : {RAW_DATA_PATH}")
    print(f"{'='*55}\n")

    df = download(
        ticker=TICKER,
        period=PERIOD,
        interval=INTERVAL,
        save=True,
        output_dir=str(PROJECT_ROOT / RAW_DATA_PATH),
    )

    print(f"\nDownload complete: {len(df)} observations")
    print(f"  Date range: {df.index[0].date()} → {df.index[-1].date()}")
    print(df.tail(5))


if __name__ == "__main__":
    main()
