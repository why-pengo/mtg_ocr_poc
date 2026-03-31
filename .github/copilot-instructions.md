# MTG OCR POC - Copilot Instructions

## Project Overview

A Python CLI proof-of-concept for reading Magic: The Gathering card names from images using OCR.
The goal is to benchmark multiple local OCR engines for accuracy against MTG card fonts, then
look up the detected card name on Scryfall via the `scrython` library.

**Tech Stack**: Python 3.10+, scrython, OpenCV/Pillow (image pre-processing), multiple OCR engines

## OCR Engines Under Test

Each engine is evaluated for accuracy on MTG card name text (stylised serif fonts, varied backgrounds):

| Engine | Package | System deps? | Isolation |
|---|---|---|---|
| Tesseract | `pytesseract` | Yes — `tesseract-ocr` | Docker |
| EasyOCR | `easyocr` | No | Native `.venv` |
| PaddleOCR | `paddlepaddle` + `paddleocr` | No | Native `.venv` |
| TrOCR | `transformers` + `torch` | No | Native `.venv` |

**Docker rule**: Any engine that requires system-level packages (e.g. Tesseract) must be wrapped
in a Docker container. Do not install system packages on the host. Native-only engines run in `.venv`.

## CLI UX Flow

```
python main.py <image_path>

→ Run all available OCR engines on the image
→ For each engine, print: engine name, detected card name, confidence (if available), elapsed time
→ After all engines run, display a ranked summary table (best accuracy estimate first)
→ Prompt user: "Look up '<best_guess>' on Scryfall? [y/n/custom name]"
→ If user cancels: exit cleanly
→ If one Scryfall match: print card details
→ If multiple matches: print numbered list with OSC 8 terminal hyperlinks to scryfall.com
```

- Use **OSC 8** escape sequences for clickable terminal hyperlinks (no web UI)
- Always show the OCR result to the user before any network call
- Scryfall lookups use `scrython` — respect Scryfall's rate limit (50–100 ms between requests)

## Build & Run Commands

```bash
# Create venv and install native dependencies
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# Run on an image
.venv/bin/python main.py path/to/card.jpg

# Run Tesseract engine via Docker
docker build -t mtg-ocr-tesseract -f docker/tesseract/Dockerfile .
docker run --rm -v "$PWD":/app mtg-ocr-tesseract path/to/card.jpg

# Run tests
.venv/bin/pytest
.venv/bin/pytest --cov=.
```

## Project Structure (evolving)

```
main.py                  # CLI entry point
engines/
  base.py                # Abstract OCREngine base class
  easyocr_engine.py
  paddleocr_engine.py
  trocr_engine.py
  tesseract_engine.py    # Invokes Docker internally
docker/
  tesseract/
    Dockerfile
    run.py               # Tesseract engine script run inside container
preprocessing/
  card_detect.py         # Card boundary detection + perspective correction (OpenCV)
  image_utils.py         # Crop name region from rectified card, contrast/denoise helpers
scryfall/
  lookup.py              # scrython wrapper: search, display results, OSC 8 links
benchmark/
  results.py             # Accuracy tracking and summary table
tests/
  fixtures/              # Sample card images and expected card names
  test_engines.py
  test_lookup.py
  test_preprocessing.py
```

## Coding Conventions

### Python Code Style
- **Line length**: 100 characters (black configured)
- **Import order**: `isort` with `profile="black"`
- **Type hints**: Required on all public functions and class methods
- **Docstrings**: Required for modules and classes; optional for obvious helpers

### Error Handling
- OCR engines must never crash the CLI — catch exceptions, log the error, and report `None` result
- Scryfall 404 (no match) is a normal outcome — display a friendly message, do not raise
- Network errors from `scrython` should be caught and shown as a clear message to the user

### Testing Conventions
- **Naming**: `test_<function>_<scenario>` (e.g., `test_easyocr_detects_card_name_from_clean_image`)
- **Fixtures**: Sample images + ground-truth card names in `tests/fixtures/`
- **Engine tests**: Mock the actual OCR call to keep tests fast; have at least one integration test per engine
- Run with `.venv/bin/pytest` (not bare `pytest`)

## Installed Skills — Project Configuration

### `polyglot-test-agent`
- Run tests with `.venv/bin/pytest` (not bare `pytest`)
- Coverage target: `--cov=.`
- Test naming: `test_<function>_<scenario>`
- Fixtures (sample card images + expected names) in `tests/fixtures/`

### `refactor`
- All OCR engines implement the `OCREngine` abstract base class in `engines/base.py`
- Keep preprocessing logic in `preprocessing/image_utils.py`, not inside engine classes
- Scryfall interaction lives entirely in `scryfall/lookup.py`

## Development Workflow

1. **Branch**: Feature branches from `main`, named `issue-{number}-{short-description}`
2. **GitHub Issue**: All work tracked by a GitHub issue; reference issue number in branch name and commits
3. **Format**: Run `black .` and `isort .` before committing
4. **Test**: Run `.venv/bin/pytest` before committing
5. **Commit**: Use conventional commit prefixes:
   - `feat:` — new engine or feature
   - `fix:` — bug fixes
   - `refactor:` — code cleanup
   - `test:` — test additions/changes
   - `docs:` — documentation updates
   - `bench:` — benchmark data or scoring changes

## Important Notes

- **Card detection + rectification**: Real-world photos need perspective correction before cropping.
  Pipeline: grayscale → blur → Canny edge detection → contour finding → largest quadrilateral →
  `cv2.getPerspectiveTransform` + `cv2.warpPerspective` to produce a flat, axis-aligned card image.
  This lives in `preprocessing/card_detect.py`.
- **MTG card name region**: The card name is in the top ~10% of the card image, left of the mana cost.
  Pre-processing should crop to this region before passing to OCR engines.
- **Font challenge**: MTG uses a stylised serif font ("Matrix Bold") for card names — engines trained on
  standard fonts (Tesseract default) will perform poorly without custom training data or fine-tuning.
- **Benchmark accuracy metric**: Normalised Levenshtein distance between OCR output and ground-truth
  card name (case-insensitive, stripped). Lower edit distance = better.
- **scrython rate limiting**: Add `time.sleep(0.1)` between Scryfall calls. The `scrython` library does
  not enforce rate limits automatically.
- **OSC 8 hyperlinks**: Format is `\033]8;;{url}\033\\{text}\033]8;;\033\\`. Test in terminals that
  support it (iTerm2, Kitty, recent GNOME Terminal). Gracefully degrade to plain URLs otherwise.
- **Docker engine wrapper**: `tesseract_engine.py` should run the Docker container as a subprocess,
  pass the image via a volume mount, and parse JSON output from stdout.

## GitHub Issue Writing Guidelines

### Bug Reports
- **Title**: Short and specific — e.g., *"EasyOCR returns empty string on high-contrast card art"*
- **Steps to reproduce**: Numbered, minimal steps to trigger the bug
- **Expected vs. actual behavior**: What should happen vs. what does happen
- **Error output**: Full tracebacks in fenced code blocks

### Feature Requests / Experiments
- **Title**: Action-oriented — e.g., *"Add PaddleOCR engine with name-region crop"*
- **Problem statement**: What does this experiment test?
- **Acceptance criteria**: Bullet list of "done when..." conditions including benchmark targets

### Acceptance Criteria Template
```
### Acceptance Criteria
- [ ] <observable outcome 1>
- [ ] <observable outcome 2>
- [ ] Tests cover the new behavior
```
