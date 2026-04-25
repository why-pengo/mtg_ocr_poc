"""Tests for the batch ingestion pipeline: osxphotos source, export, and Scryfall lookup."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ingestion.export import write_json
from scryfall.lookup import lookup_for_ingestion


# ---------------------------------------------------------------------------
# export.write_json
# ---------------------------------------------------------------------------


class TestWriteJson:
    def test_creates_valid_json_file(self, tmp_path: Path) -> None:
        cards = [{"id": "abc123", "name": "Lightning Bolt", "set": "lea"}]
        out = tmp_path / "cards.json"
        write_json(cards, out)
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded == cards

    def test_writes_array_of_dicts(self, tmp_path: Path) -> None:
        cards = [
            {"id": "aaa", "name": "Black Lotus"},
            {"id": "bbb", "name": "Counterspell"},
        ]
        out = tmp_path / "out.json"
        write_json(cards, out)
        loaded = json.loads(out.read_text())
        assert len(loaded) == 2
        assert loaded[0]["name"] == "Black Lotus"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "deep" / "cards.json"
        write_json([{"id": "x", "name": "Bolt"}], out)
        assert out.exists()

    def test_empty_list_writes_empty_array(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.json"
        write_json([], out)
        assert json.loads(out.read_text()) == []

    def test_non_ascii_card_names_preserved(self, tmp_path: Path) -> None:
        cards = [{"id": "zzz", "name": "Jötun Grunt"}]
        out = tmp_path / "cards.json"
        write_json(cards, out)
        loaded = json.loads(out.read_text())
        assert loaded[0]["name"] == "Jötun Grunt"


# ---------------------------------------------------------------------------
# ingestion.osxphotos_source.iter_album_photos
# ---------------------------------------------------------------------------


class TestIterAlbumPhotos:
    def test_raises_runtime_error_on_non_macos(self) -> None:
        from ingestion.osxphotos_source import iter_album_photos

        with patch.object(sys, "platform", "linux"):
            with pytest.raises(RuntimeError, match="macOS"):
                list(iter_album_photos("MTG Cards to Scan"))

    @pytest.mark.skipif(sys.platform != "darwin", reason="osxphotos is macOS-only")
    def test_raises_value_error_for_missing_album(self) -> None:
        mock_db = MagicMock()
        mock_db.album_info = []

        with patch("ingestion.osxphotos_source.sys") as mock_sys:
            mock_sys.platform = "darwin"
            with patch.dict("sys.modules", {"osxphotos": MagicMock(PhotosDB=lambda: mock_db)}):
                from importlib import reload

                import ingestion.osxphotos_source as src

                reload(src)
                with pytest.raises(ValueError, match="not found"):
                    list(src.iter_album_photos("Nonexistent Album"))

    @pytest.mark.skipif(sys.platform != "darwin", reason="osxphotos is macOS-only")
    def test_skips_photos_not_on_disk(self, tmp_path: Path, capsys) -> None:
        mock_photo = MagicMock()
        mock_photo.filename = "card.jpg"
        mock_photo.path = None  # not on disk

        mock_album = MagicMock()
        mock_album.title = "MTG Cards to Scan"
        mock_album.photos = [mock_photo]

        mock_db = MagicMock()
        mock_db.album_info = [mock_album]

        mock_osxphotos = MagicMock()
        mock_osxphotos.PhotosDB.return_value = mock_db

        with patch.dict("sys.modules", {"osxphotos": mock_osxphotos}):
            from importlib import reload

            import ingestion.osxphotos_source as src

            reload(src)
            results = list(src.iter_album_photos("MTG Cards to Scan"))

        assert results == []
        captured = capsys.readouterr()
        assert "Skipping" in captured.out

    @pytest.mark.skipif(sys.platform != "darwin", reason="osxphotos is macOS-only")
    def test_yields_filename_and_ndarray(self, tmp_path: Path) -> None:
        # Write a tiny real image so cv2.imread can read it
        import cv2

        img_path = tmp_path / "card.jpg"
        img = np.zeros((100, 70, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        mock_photo = MagicMock()
        mock_photo.filename = "card.jpg"
        mock_photo.path = str(img_path)

        mock_album = MagicMock()
        mock_album.title = "MTG Cards to Scan"
        mock_album.photos = [mock_photo]

        mock_db = MagicMock()
        mock_db.album_info = [mock_album]

        mock_osxphotos = MagicMock()
        mock_osxphotos.PhotosDB.return_value = mock_db

        with patch.dict("sys.modules", {"osxphotos": mock_osxphotos}):
            from importlib import reload

            import ingestion.osxphotos_source as src

            reload(src)
            results = list(src.iter_album_photos("MTG Cards to Scan"))

        assert len(results) == 1
        filename, image = results[0]
        assert filename == "card.jpg"
        assert isinstance(image, np.ndarray)
        assert image.shape == (100, 70, 3)


# ---------------------------------------------------------------------------
# scryfall.lookup.lookup_for_ingestion
# ---------------------------------------------------------------------------


class TestLookupForIngestion:
    def test_returns_card_dict_on_named_match(self) -> None:
        card_data = {
            "id": "abc123",
            "name": "Lightning Bolt",
            "set": "lea",
            "set_name": "Limited Edition Alpha",
            "scryfall_uri": "https://scryfall.com/card/lea/161",
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = card_data

        with patch("scryfall.lookup.requests.get", return_value=mock_resp):
            result = lookup_for_ingestion("Lightning Bolt")

        assert result == card_data

    def test_returns_none_when_no_match(self) -> None:
        not_found = MagicMock()
        not_found.status_code = 404

        with patch("scryfall.lookup.requests.get", return_value=not_found):
            result = lookup_for_ingestion("Zzz Fake Card 9999")

        assert result is None

    def test_falls_through_to_search_on_named_failure(self, monkeypatch) -> None:
        card_data = {
            "id": "bbb",
            "name": "Bolt",
            "set": "lea",
            "set_name": "Alpha",
            "scryfall_uri": "https://scryfall.com/card/lea/161",
        }
        named_resp = MagicMock()
        named_resp.status_code = 404

        search_resp = MagicMock()
        search_resp.status_code = 200
        search_resp.json.return_value = {"data": [card_data]}

        responses = [named_resp, search_resp]

        with patch("scryfall.lookup.requests.get", side_effect=responses):
            # Auto-pick option 1
            monkeypatch.setattr("builtins.input", lambda _: "1")
            result = lookup_for_ingestion("Bolt")

        assert result == card_data

    def test_user_can_skip_disambiguation(self, monkeypatch) -> None:
        named_resp = MagicMock()
        named_resp.status_code = 404

        search_resp = MagicMock()
        search_resp.status_code = 200
        search_resp.json.return_value = {
            "data": [
                {"id": "1", "name": "Card A", "set": "a", "set_name": "Alpha",
                 "scryfall_uri": "https://scryfall.com/1"},
                {"id": "2", "name": "Card B", "set": "b", "set_name": "Beta",
                 "scryfall_uri": "https://scryfall.com/2"},
            ]
        }

        with patch("scryfall.lookup.requests.get", side_effect=[named_resp, search_resp]):
            monkeypatch.setattr("builtins.input", lambda _: "s")
            result = lookup_for_ingestion("Card")

        assert result is None
