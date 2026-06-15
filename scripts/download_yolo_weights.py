from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path


HF_BASE_URL = "https://huggingface.co/Ultralytics/YOLO11/resolve/main"
KNOWN_WEIGHTS = {
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt",
}
SHA256 = {
    "yolo11m.pt": "d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download public YOLO11 pretrained weights before offline training."
    )
    parser.add_argument("--name", default="yolo11m.pt", choices=sorted(KNOWN_WEIGHTS))
    parser.add_argument("--out-dir", type=Path, default=Path("weights"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def download(url: str, target: Path, force: bool = False) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        print(target)
        return target

    tmp = target.with_suffix(target.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as fp:
        shutil.copyfileobj(response, fp)
    tmp.replace(target)
    expected_hash = SHA256.get(target.name)
    if expected_hash:
        actual_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            target.unlink(missing_ok=True)
            raise RuntimeError(
                f"checksum mismatch for {target.name}: expected {expected_hash}, got {actual_hash}"
            )
    print(target)
    return target


def main() -> None:
    args = parse_args()
    url = f"{HF_BASE_URL}/{args.name}"
    download(url, args.out_dir / args.name, force=args.force)


if __name__ == "__main__":
    main()
