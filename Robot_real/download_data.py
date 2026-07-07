"""Download and extract the real-robot datasets from Google Drive."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

try:
    import gdown
except ImportError as exc:
    raise SystemExit(
        "This script requires gdown. Install it with: pip install gdown"
    ) from exc

DATASETS = {
    "peg_in_hole_camera": "11_K9_jZUrj5Wn-nnJtWA8v6vHFIwvLDH",
    "peg_in_hole_trajectory": "1aHehI6P7tjiDkTdqMkKj3KyYvKo9Tpse",
    "sweep_camera": "1bjXfy5txVUPq9vyuppslVRAiKbymANiE",
    "sweep_trajectory": "1KHhyPZhlswp_TLRq3hiKGpHJenCgHmUW",
    "pick_and_place_camera": "1Xa5j_zUzNpjB4kzX36gm-g4EHsY5SM2w",
    "pick_and_place_trajectory": "1vDbzkEw-X720sglMz3HcZHISi1G5f_jJ",
}

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "real_dataset"


def _safe_extract(archive: Path, destination: Path) -> None:
    """Extract an archive while preventing paths from escaping destination."""
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            member_path = (destination / member.filename).resolve()
            if not member_path.is_relative_to(destination_resolved):
                raise ValueError(f"Unsafe path in {archive.name}: {member.filename}")
        zip_file.extractall(destination)


def download_dataset(name: str, file_id: str, output_dir: Path, force: bool) -> None:
    destination = output_dir / name
    if destination.exists() and not force:
        print(f"Skipping {name}: {destination} already exists (use --force to replace it).")
        return

    archive = output_dir / f".{name}.zip"
    partial_archive = output_dir / f".{name}.zip.part"

    if force and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Downloading {name}...")
        partial_archive.unlink(missing_ok=True)
        result = gdown.download(id=file_id, output=str(partial_archive), quiet=False)
        if result is None:
            raise RuntimeError(f"Google Drive download failed for {name}")
        partial_archive.replace(archive)

        print(f"Extracting {name} to {destination}...")
        _safe_extract(archive, destination)
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    finally:
        archive.unlink(missing_ok=True)
        partial_archive.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Extraction directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace dataset directories that already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, file_id in DATASETS.items():
        download_dataset(name, file_id, output_dir, args.force)

    print(f"All datasets are available in {output_dir}")


if __name__ == "__main__":
    main()
