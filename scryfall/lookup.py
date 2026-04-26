"""Scryfall card lookup via scrython with OSC 8 terminal hyperlinks."""

from __future__ import annotations

import time
from typing import Any, Optional

import requests
import scrython

_SCRYFALL_NAMED_URL = "https://api.scryfall.com/cards/named"
_SCRYFALL_SEARCH_URL = "https://api.scryfall.com/cards/search"

_RATE_LIMIT_S = 0.1  # 100 ms between calls per Scryfall policy


def _hyperlink(url: str, text: str) -> str:
    """Wrap *text* in an OSC 8 terminal hyperlink pointing to *url*.

    Renders as a clickable link in iTerm2, Kitty, GNOME Terminal 3.26+, etc.
    Degrades gracefully to plain text in unsupported terminals.
    """
    return f"\033]8;;{url}\033\\{text}\033]8;;\033\\"


def _rate_limit() -> None:
    time.sleep(_RATE_LIMIT_S)


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
        link = _hyperlink(url, card["name"])
        print(f"  {i}. {link}  —  {set_text}")


def _print_single_card(card: object) -> None:
    url = card.scryfall_uri()  # type: ignore[attr-defined]
    link = _hyperlink(url, card.name())  # type: ignore[attr-defined]
    print(f"\n  {link}")
    print(f"  {card.type_line()}")  # type: ignore[attr-defined]
    print(f"  {card.set_name()} ({card.set().upper()})")  # type: ignore[attr-defined]


def scryfall_search(name: str) -> list[dict[str, Any]]:
    """Search Scryfall for *name*; return a list of card dicts (empty on failure).

    Tries fuzzy named lookup first; falls back to broader search.
    Never raises — returns [] on any error.
    """
    _rate_limit()

    try:
        resp = requests.get(
            _SCRYFALL_NAMED_URL,
            params={"fuzzy": name},
            timeout=10,
        )
        if resp.ok:
            card = resp.json()
            return [card]
    except Exception:
        pass

    _rate_limit()

    try:
        resp = requests.get(
            _SCRYFALL_SEARCH_URL,
            params={"q": f'name:"{name}"'},
            timeout=10,
        )
        if resp.ok:
            data = resp.json()
            return data.get("data", [])
    except Exception:
        pass

    return []


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
        resp = requests.get(_SCRYFALL_NAMED_URL, params={"fuzzy": name}, timeout=10)
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
        link = _hyperlink(url, card["name"]) if url else card["name"]
        print(f"  ✓  {link}  —  {card.get('set_name', '')} ({card.get('set', '').upper()})")
        return card

    _rate_limit()
    try:
        search_resp = requests.get(
            _SCRYFALL_SEARCH_URL, params={"q": f'name:"{name}"'}, timeout=10
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
        link = _hyperlink(url, card["name"]) if url else card["name"]
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
