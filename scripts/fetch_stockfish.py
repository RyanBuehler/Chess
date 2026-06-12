"""Provision the pinned Stockfish binary into tools/stockfish/ (gitignored).

Downloads the official Stockfish 17.1 Windows AVX2 release from GitHub, extracts
it, and normalizes the engine to tools/stockfish/stockfish.exe so
default_stockfish_path() discovers it. The exact release URL is pinned so anchor
UCI_Elo calibration is reproducible (a different build silently moves anchors).

Usage:  python scripts/fetch_stockfish.py
        python scripts/fetch_stockfish.py --url <override>   # e.g. a Linux build
"""
import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

# Pinned Stockfish 17.1 Windows x86-64-avx2 build.
DEFAULT_URL = (
    "https://github.com/official-stockfish/Stockfish/releases/download/"
    "sf_17.1/stockfish-windows-x86-64-avx2.zip"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def fetch(url: str = DEFAULT_URL) -> Path:
    dest = _repo_root() / "tools" / "stockfish"
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "download.zip"
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, archive)

    print(f"extracting {archive}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)

    # Find the extracted engine executable and normalize its name.
    candidates = [
        p for p in dest.rglob("stockfish*")
        if p.is_file() and (p.suffix.lower() in (".exe", "") and "download" not in p.name)
    ]
    if not candidates:
        raise SystemExit("no stockfish executable found in the archive")
    exe = max(candidates, key=lambda p: p.stat().st_size)   # the real binary is large
    target = dest / ("stockfish.exe" if exe.suffix.lower() == ".exe" or sys.platform == "win32" else "stockfish")
    if exe != target:
        shutil.copy2(exe, target)
    if sys.platform != "win32":
        target.chmod(0o755)
    archive.unlink(missing_ok=True)
    print(f"stockfish ready at {target}")
    return target


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    args = ap.parse_args(argv)
    fetch(args.url)


if __name__ == "__main__":
    main()
