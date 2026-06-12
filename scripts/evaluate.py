"""Elo evaluator entry point.

Once over all runs:  python scripts/evaluate.py --once --runs-root runs
Watch daemon:        python scripts/evaluate.py --runs-root runs --config experiments/eval.yaml
Restrict to one run: python scripts/evaluate.py --once --run <run-id>
Stop the daemon:     create runs/EVAL_STOP
"""
import sys

from chessrl.evaluation.daemon import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]) or 0)
