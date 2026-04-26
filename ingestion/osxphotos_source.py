"""Apple Photos album reader via osxphotos (macOS only)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


def iter_album_photos(album_name: str) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``(filename, BGR ndarray)`` for each photo in a named Apple Photos album.

    Photos that are not downloaded to disk (e.g. iCloud-only) are skipped with a warning.

    Raises:
        RuntimeError: if called on a non-macOS platform or if ``osxphotos`` is not installed.
        ValueError: if no album with the given name exists in the Photos library.
    """
    if sys.platform != "darwin":
        raise RuntimeError(
            "iter_album_photos requires macOS — osxphotos is not available on this platform."
        )

    try:
        import osxphotos
    except ImportError as exc:
        raise RuntimeError("osxphotos is not installed. Run: pip install osxphotos") from exc

    db = osxphotos.PhotosDB()
    album = next((a for a in db.album_info if a.title == album_name), None)
    if album is None:
        available = [a.title for a in db.album_info]
        raise ValueError(
            f"Apple Photos album '{album_name}' not found.\n" f"Available albums: {available}"
        )

    photos = album.photos
    if not photos:
        print(f"  Album '{album_name}' is empty.")
        return

    for photo in photos:
        path = photo.path
        if not path or not Path(path).exists():
            print(
                f"  ⚠️  Skipping {photo.filename} — not on disk "
                f"(ensure iCloud download is complete)"
            )
            continue

        image = cv2.imread(path)
        if image is None:
            print(f"  ⚠️  Could not read {photo.filename} — skipping")
            continue

        yield photo.filename, image
