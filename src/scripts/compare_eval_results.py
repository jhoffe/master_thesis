from pathlib import Path
import json
import re
import pandas as pd
from typing import Dict, Any

from utils.compare_results import METRICS_DIR, FNAME_RE, load_metrics_flat, compare_eval_results

def main(metrics_dir: Path, regex: re.Pattern) -> None:
    df = compare_eval_results(metrics_dir, regex)
    df.to_csv("reports/metrics/metrics_flat_all.csv", index=False)

if __name__ == "__main__":
    main(METRICS_DIR, FNAME_RE)