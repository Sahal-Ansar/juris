"""Compare experiments from eval/results/ (PLAN 5.5).

    uv run scripts/compare_experiments.py EXP [EXP ...]
        a Markdown table: each headline metric's mean over items with its 95% CI
    uv run scripts/compare_experiments.py A B --paired
        also a paired comparison of B against A on the items both ran (bootstrap CI of the
        mean difference, sign-flip randomisation p-value)

Options: --metrics NAME [NAME ...] (default: the headline metrics), --results-dir DIR,
--out FILE (write the Markdown there as well as printing it).
"""

import argparse
import sys
from pathlib import Path

from juris.eval.compare import comparison_table, load_experiment, paired_table
from juris.eval.runner import HEADLINE, RESULTS_DIR


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("experiments", nargs="+")
    ap.add_argument("--paired", action="store_true", help="paired test of the 2nd vs the 1st")
    ap.add_argument("--metrics", nargs="+", default=list(HEADLINE))
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    if args.paired and len(args.experiments) != 2:
        ap.error("--paired needs exactly two experiments")

    experiments = [load_experiment(args.results_dir / e) for e in args.experiments]
    out = comparison_table(experiments, args.metrics)
    if args.paired:
        out += "\n\n" + paired_table(experiments[0], experiments[1], args.metrics)
    print(out)
    if args.out:
        args.out.write_text(out + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
