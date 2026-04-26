"""Scryfall card lookup via scrython with OSC 8 terminal hyperlinks."""

from __future__ import annotations

import time
from typing import Any, Optional

import requests
import scrython

_SCRYFALL_NAMED_URL = "https://api.scryfall.com/cards/named"
_SCRYFALL_SEARCH_URL = "https://api.scryfall.com/cards/search"

_RATE_LIMIT_S = 0.1  # 100 ms between calls per Scryfall policy
_HEADERS = {
    "User-Agent": "mtg-ocr-poc/1.0 (github.com/why-pengo/mtg_ocr_poc)",
    "Accept": "application/json;q=0.9,*/*;q=0.8",
}


def hyperlink(url: str, text: str) -> str:
    """Wrap *text* in an OSC 8 terminal hyperlink pointing to *url*.

    Renders as a clickable link in iTerm2, Kitty, GNOME Terminal 3.26+, etc.
    Degrades gracefully to plain text in unsupported terminals.
    """
    return f"\033]8;;{url}\033\\{text}\033]8;;\033\\"


# Backward-compatible alias used by the stdout CLI helpers below.
_hyperlink = hyperlink


def _rate_limit() -> None:
    time.sleep(_RATE_LIMIT_S)


def _scryfall_get_json(
    url: str, params: dict[str, str]
) -> tuple[Optional[dict[str, Any]], Optional[str], Optional[int]]:
    """GET *url* with *params* and return ``(payload, error_message, status_code)``.

    Returns ``(None, error_message, status_code)`` on any failure so callers
    can decide whether to surface the error or fall through to a search.
    """
    try:
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=10)
    except requests.RequestException as exc:
        return None, f"Scryfall request failed: {exc}", None

    if resp.status_code == 429:
        return None, "Scryfall rate limit exceeded — please wait and retry.", resp.status_code
    if 500 <= resp.status_code < 600:
        return (
            None,
            f"Scryfall server error ({resp.status_code}) — please retry later.",
            resp.status_code,
        )
    if not resp.ok:
        return None, f"Scryfall returned HTTP {resp.status_code}.", resp.status_code

    try:
        return resp.json(), None, resp.status_code
    except ValueError:
        return None, "Scryfall returned invalid JSON.", resp.status_code


def prompt_and_lookup(detected_name: str) -> None:
    """Show the detected name to the user, confirm, then run the Scryfall lookup."""
    print(f"\n🃏  Detected card name: \033[1m{detected_name}\033[0m")
    response = input("Look up on Scryfall? [Y/n/custom name] ").strip()

    if response.lower() == "n":
        print("Cancelled.")
        return

    query = response if response and response.lower() not in ("y", "") else detected_name
    _lookup_and_display(query)


def _lookup_and_display(name: str) -> None:
    """Run a Scryfall search for *name* and display the result(s)."""
    print(f'\nSearching Scryfall for "{name}"…')
    _rate_limit()

    # Attempt a fuzzy named match first (returns a single best-match card)
    try:
        card = scrython.cards.Named(fuzzy=name)
        _print_single_card(card)
        return
    except Exception:
        pass  # fall through to broader search

    _rate_limit()
    try:
        search = scrython.cards.Search(q=f'name:"{name}"')
        cards = search.data()  # list of card dicts
    except Exception:
        print(f'No cards found matching "{name}".')
        return

    if not cards:
        print(f'No cards found matching "{name}".')
        return

    print(f"\nFound {len(cards)} match(es):\n")
    for i, card in enumerate(cards, 1):
        url = card["scryfall_uri"]
        set_text = f"{card['set_name']} ({card['set'].upper()})"
        link = hyperlink(url, card["name"])
        print(f"  {i}. {link}  —  {set_text}")


def _print_single_card(card: object) -> None:
    url = card.scryfall_uri()  # type: ignore[attr-defined]
    link = hyperlink(url, card.name())  # type: ignore[attr-defined]
    print(f"\n  {link}")
    print(f"  {card.type_line()}")  # type: ignore[attr-defined]
    print(f"  {card.set_name()} ({card.set().upper()})")  # type: ignore[attr-defined]


def scryfall_search(name: str) -> tuple[list[dict[str, Any]], Optional[str]]:
    """Search Scryfall for *name* and return ``(cards, error_message)``.

    Tries fuzzy named lookup first; falls back to broader search.
    Returns ``([], None)`` when no cards match, and ``([], error_message)``
    for network/HTTP/JSON failures.
    Never raises.
    """
    _rate_limit()

    card, error, status = _scryfall_get_json(_SCRYFALL_NAMED_URL, {"fuzzy": name})
    if card is not None:
        return [card], None
    # 404 means "no exact match" — fall through to search; other errors propagate
    if error is not None and status != 404:
        return [], error

    _rate_limit()

    data, error, _ = _scryfall_get_json(_SCRYFALL_SEARCH_URL, {"q": f'name:"{name}"'})
    if error is not None:
        return [], error

    cards: list[dict[str, Any]] = data.get("data", []) if data is not None else []
    return cards, None


def lookup_for_ingestion(name: str) -> Optional[dict[str, Any]]:
    """Fuzzy-search Scryfall and return a raw card dict for batch ingestion.

    Performs a fuzzy named lookup first; falls back to a broader search if that fails.
    When multiple results are returned, prompts the user to pick one interactively.

    The returned dict is a Scryfall API card object — compatible with
    ``PaperCard.upsert_from_scryfall()`` in the mtgas app.

    Returns None if no card is found or the user skips disambiguation.
    """
    print(f'  🔎  Searching Scryfall for "{name}"…')
    _rate_limit()

    try:
        resp = requests.get(
            _SCRYFALL_NAMED_URL, params={"fuzzy": name}, headers=_HEADERS, timeout=10
        )
    except requests.exceptions.RequestException as exc:
        print(f"  ✗  Network error contacting Scryfall: {exc}")
        return None

    if resp.status_code == 200:
        try:
            card = resp.json()
        except ValueError:
            print("  ✗  Invalid response from Scryfall.")
            return None
        url = card.get("scryfall_uri", "")
        link = hyperlink(url, card["name"]) if url else card["name"]
        print(f"  ✓  {link}  —  {card.get('set_name', '')} ({card.get('set', '').upper()})")
        return card

    _rate_limit()
    try:
        search_resp = requests.get(
            _SCRYFALL_SEARCH_URL, params={"q": f'name:"{name}"'}, headers=_HEADERS, timeout=10
        )
    except requests.exceptions.RequestException as exc:
        print(f"  ✗  Network error contacting Scryfall: {exc}")
        return None

    if search_resp.status_code == 429:
        print("  ✗  Scryfall rate limit hit — please wait and retry.")
        return None
    if search_resp.status_code >= 500:
        print(f"  ✗  Scryfall server error ({search_resp.status_code}) — please retry later.")
        return None
    if search_resp.status_code != 200:
        print(f'  ✗  No cards found matching "{name}".')
        return None

    try:
        cards: list[dict[str, Any]] = search_resp.json().get("data", [])
    except ValueError:
        print("  ✗  Invalid response from Scryfall.")
        return None
    if not cards:
        print(f'  ✗  No cards found matching "{name}".')
        return None

    print(f"\n  Found {len(cards)} match(es):\n")
    for i, card in enumerate(cards, 1):
        url = card.get("scryfall_uri", "")
        set_text = f"{card.get('set_name', '')} ({card.get('set', '').upper()})"
        link = hyperlink(url, card["name"]) if url else card["name"]
        print(f"    {i}. {link}  —  {set_text}")

    while True:
        choice = input(f"\n  Pick [1–{len(cards)}] or [s]kip: ").strip().lower()
        if choice == "s":
            return None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(cards):
                return cards[idx]
        except ValueError:
            pass
        print(f"  Please enter a number between 1 and {len(cards)}.")
