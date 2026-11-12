"""Repository paths shared by the package and the scripts."""

from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
CONFIG = REPO_DIR / "configs" / "benchmark.yaml"
RESULTS_DIR = REPO_DIR / "results" / "erlanger_traeger_benchmark"
FIGURES_DIR = REPO_DIR / "results" / "paper" / "figures"
TABLES_DIR = REPO_DIR / "results" / "paper" / "tables"
