"""Tests for Scryfall lookup and OSC 8 hyperlink helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from scryfall.lookup import _hyperlink, _lookup_and_display


class TestHyperlink:
    def test_contains_url_and_text(self) -> None:
        result = _hyperlink("https://scryfall.com/card/abc", "Lightning Bolt")
        assert "https://scryfall.com/card/abc" in result
        assert "Lightning Bolt" in result

    def test_uses_osc8_escape_sequences(self) -> None:
        result = _hyperlink("https://example.com", "Test")
        assert "\033]8;;" in result
        assert "\033\\" in result


class TestLookupAndDisplay:
    def test_prints_card_details_on_single_match(self, capsys) -> None:
        mock_card = MagicMock()
        mock_card.name.return_value = "Lightning Bolt"
        mock_card.type_line.return_value = "Instant"
        mock_card.set_name.return_value = "Limited Edition Alpha"
        mock_card.set.return_value = "lea"
        mock_card.scryfall_uri.return_value = "https://scryfall.com/card/lea/161"

        with patch("scrython.cards.Named", return_value=mock_card):
            _lookup_and_display("Lightning Bolt")

        captured = capsys.readouterr()
        assert "Lightning Bolt" in captured.out

    def test_falls_through_to_search_on_named_failure(self, capsys) -> None:
        mock_search = MagicMock()
        mock_search.data.return_value = [
            {
                "name": "Bolt",
                "set_name": "Alpha",
                "set": "lea",
                "scryfall_uri": "https://scryfall.com/card/lea/161",
            }
        ]

        with patch("scrython.cards.Named", side_effect=Exception("not found")):
            with patch("scrython.cards.Search", return_value=mock_search):
                _lookup_and_display("Bolt")

        captured = capsys.readouterr()
        assert "Bolt" in captured.out

    def test_prints_no_match_message_when_both_fail(self, capsys) -> None:
        with patch("scrython.cards.Named", side_effect=Exception("not found")):
            with patch("scrython.cards.Search", side_effect=Exception("not found")):
                _lookup_and_display("Zzz Fake Card Xyz 9999")

        captured = capsys.readouterr()
        assert "No cards found" in captured.out
