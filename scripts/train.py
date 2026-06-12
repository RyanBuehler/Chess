"""Entry point.

Single process (smoke / small): python scripts/train.py --config experiments/foo.yaml
Parallel self-play (M5):         python scripts/train.py --parallel --config experiments/foo.yaml --games 200
"""
import sys


def main() -> None:
    argv = sys.argv[1:]
    if "--parallel" in argv:
        argv = [a for a in argv if a != "--parallel"]
        from chessrl.training.parallel_loop import main as parallel_main

        parallel_main(argv)
    else:
        from chessrl.training.loop import main as loop_main

        loop_main(argv)


if __name__ == "__main__":
    main()
