# =============================================================================
# scripts/process_data.py
#
# Orchestration only — contains NO mathematics.
#
# Pipeline:
#   load raw CSV  →  clean  →  print report  →  save D_clean
# =============================================================================

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import TICKER, INTERVAL, RAW_DATA_PATH, PROCESSED_DATA_PATH
from src.data.loader import load_latest
from src.data.cleaner import clean, save_cleaned, print_report


def main() -> None:
    print(f"\n{'='*55}")
    print(f"  LS-HFT V0 — Process Data  (D_raw → D_clean)")
    print(f"{'='*55}\n")

    # Load
    raw_dir = str(PROJECT_ROOT / RAW_DATA_PATH)
    df_raw = load_latest(TICKER, INTERVAL, raw_dir)

    print(f"Raw data loaded: {len(df_raw)} rows")

    # Clean — all transformation logic is inside cleaner.py
    df_clean, report = clean(df_raw)

    # Transparent report — never hide what was removed
    print_report(report)

    # Save
    out_dir = str(PROJECT_ROOT / PROCESSED_DATA_PATH)
    path = save_cleaned(df_clean, TICKER, INTERVAL, out_dir)

    print(f"Cleaned data: {len(df_clean)} rows saved → {path}")


if __name__ == "__main__":
    main()
