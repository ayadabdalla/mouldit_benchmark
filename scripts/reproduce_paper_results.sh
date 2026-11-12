#!/usr/bin/env bash
# Every figure and table of the paper from the benchmark results, into results/paper/.
#   bash scripts/reproduce_paper_results.sh        # figures and tables from existing results
#   bash scripts/reproduce_paper_results.sh --run  # first run (or resume) the full benchmark
set -euo pipefail
cd "$(dirname "$0")/.."

python -m mouldit_benchmark.datasets.download_reference_cycle
if [[ "${1:-}" == "--run" ]]; then
    python scripts/run_benchmark.py
fi
python scripts/paper/fig_reference_cycle.py  # Fig. 2 and the phase marker table
python scripts/paper/fig_synthetic_data.py  # Figs. 3 and 4
python scripts/paper/fig_results.py  # Figs. 6-9
python scripts/paper/fig_latent.py  # Fig. 10
python scripts/paper/table_results.py  # Tables II and IV
python scripts/paper/table_latent.py  # Table III and the numbers of the Latent Analysis section
