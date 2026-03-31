"""Scryfall card lookup via scrython with OSC 8 terminal hyperlinks."""
from __future__ import annotations

import time

import scrython

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
