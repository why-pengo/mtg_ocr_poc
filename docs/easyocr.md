# EasyOCR: How We Use It and Why

This document explains what EasyOCR is, how it fits into this project's pipeline,
every tuning decision made, and where to go to learn more.

---

## Table of Contents

1. [What is EasyOCR?](#what-is-easyocr)
2. [The Full Pipeline](#the-full-pipeline)
3. [Step 1 — Card Detection & Perspective Correction](#step-1--card-detection--perspective-correction)
4. [Step 2 — Name Region Crop](#step-2--name-region-crop)
5. [Step 3 — Image Enhancement](#step-3--image-enhancement)
6. [Step 4 — EasyOCR Recognition](#step-4--easyocr-recognition)
7. [Step 5 — Text Normalisation](#step-5--text-normalisation)
8. [Accuracy Scoring](#accuracy-scoring)
9. [Known Limitations](#known-limitations)
10. [Tuning Levers](#tuning-levers)
11. [Further Reading](#further-reading)

---

## What is EasyOCR?

EasyOCR is a Python library that wraps two deep-learning models into a single `reader.readtext()` call:

| Stage | Model | What it does |
|---|---|---|
| **Detection** | [CRAFT](https://arxiv.org/abs/1904.01941) | Finds *where* text is in the image — outputs bounding boxes around each word/character cluster |
| **Recognition** | [CRNN](https://arxiv.org/abs/1507.05717) | Reads *what* the text says inside each bounding box |

You can think of CRAFT as "finding the text" and CRNN as "reading the text". They're two separate
neural networks chained together.

**Why EasyOCR first?**

- No system packages required (unlike Tesseract, which needs `apt-get install tesseract-ocr`)
- Reasonably good out-of-the-box accuracy on stylised fonts
- Simple API: one call returns `(bounding_box, text_string, confidence_score)` triples
- Active maintenance, good documentation

Models are downloaded on first use (~100 MB) and cached in `~/.EasyOCR/`.

> **Further reading:** [EasyOCR GitHub](https://github.com/JaidedAI/EasyOCR) ·
> [CRAFT paper](https://arxiv.org/abs/1904.01941) ·
> [CRNN paper](https://arxiv.org/abs/1507.05717)

---

## The Full Pipeline

Before EasyOCR ever sees a pixel, the image passes through several preprocessing steps.
This matters a lot — garbage in, garbage out.

```
Photo of card (JPEG)
        │
        ▼
┌─────────────────────────────┐
│  Step 1: Card detection     │  Find the card rectangle, fix the angle
│  & perspective correction   │  preprocessing/card_detect.py
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Step 2: Name region crop   │  Cut out only the top strip of the card
│                             │  preprocessing/image_utils.py
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Step 3: Image enhancement  │  Grayscale + CLAHE + denoise
│                             │  preprocessing/image_utils.py
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Step 4: EasyOCR            │  CRAFT detection → CRNN recognition
│                             │  engines/easyocr_engine.py
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Step 5: Text normalisation │  Fix common OCR character confusions
│                             │  preprocessing/image_utils.py
└─────────────────────────────┘
        │
        ▼
   Card name string
```

---

## Step 1 — Card Detection & Perspective Correction

**File:** `preprocessing/card_detect.py`

When you photograph a card on a table, two things are almost always true:

1. The card is tilted at some angle (not perfectly aligned with the camera).
2. There may be other objects in the frame (table surface, other cards, packaging).

We need to find the card, isolate it, and "flatten" it to a clean top-down rectangle
before any cropping can happen. This is called a **perspective transform** (or homography).

### Strategy A — Outer edge detection (standard bordered cards)

This is the main strategy. The approach:

1. **Convert to grayscale** — colour isn't needed for edge detection.
2. **Bilateral filter** — a blur that *preserves* edges while removing noise.
   We use this instead of a plain Gaussian blur because Gaussian smooths edges away,
   making them harder to detect. Bilateral filter smooths flat areas but leaves sharp
   transitions (like a card border against a table) intact.
3. **Canny edge detection** — finds the outlines of objects by looking for sharp
   changes in pixel brightness. Returns a binary image (edges = white, everything else = black).
4. **Dilate the edges** — expand edge pixels by 1 pixel so small gaps in the card border close up.
5. **Find contours** — trace the connected edge regions into shapes.
6. **Pick the largest quadrilateral** that has roughly a card-shaped aspect ratio (63:88mm ≈ 1:1.4).
7. **Perspective warp** — `cv2.getPerspectiveTransform` + `cv2.warpPerspective` maps the
   four detected corner points to a perfectly rectangular output image.

We retry Canny with three progressively lower threshold pairs before giving up:

| Attempt | Low threshold | High threshold | Good for |
|---|---|---|---|
| 1 | 50 | 150 | High-contrast backgrounds |
| 2 | 30 | 100 | Medium-contrast backgrounds |
| 3 | 15 | 60 | Low-contrast / dark backgrounds |

> **Further reading:** [Canny Edge Detection explained](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html) ·
> [Bilateral Filter](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed) ·
> [Perspective Transform tutorial](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html)

### Strategy B — Inner structure detection (borderless / extended-art cards)

Some modern MTG cards have **borderless frames** where the artwork bleeds all the way to the
card edge. There is no rectangular coloured border to detect — the card edge is just dark art
transitioning to a dark table. Strategy A fails here.

Every MTG card — regardless of frame style — has a **type line**: a horizontal bar roughly
57.5% down the card that separates the artwork from the rules text. It's always there, and it
always creates a strong horizontal edge.

We detect it with a **Hough line transform**:

1. Apply a horizontal Sobel filter — this highlights horizontal edges (changes in brightness
   going top-to-bottom), which is exactly what the type line produces.
2. Threshold the result to get a binary image.
3. Restrict the search to a band between y = 47.5% and 67.5% of image height.
4. Run `cv2.HoughLinesP` — probabilistic Hough transform for line segments.
   A line must span at least 30% of the image width to be accepted.
5. Take the median y-position of all detected lines — that's the type line.
6. Back-calculate the card's full bounding box using the known proportions:
   type line is at 57.5% of card height, and the card is 1.397× taller than it is wide.

> **Further reading:** [Hough Line Transform](https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html) ·
> [Sobel Derivatives](https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html)

---

## Step 2 — Name Region Crop

**File:** `preprocessing/image_utils.py` → `crop_name_region()`

After perspective correction, we have a flat, upright card image.
Passing the whole card to EasyOCR would make it work harder and risk picking up
text from the rules box, flavour text, or artist credits.

Instead, we crop to just the name bar — a strip at the top of the card:

```
┌──────────────────────────────────┐  ← 0% (top of card)
│                                  │
│  3.4%                            │  ← top of name bar
│  ┌──────────────────────────┐    │
│  │   C A R D   N A M E      │    │  ← the crop we pass to OCR
│  └──────────────────────────┘    │  ← 11.5% (bottom of name bar)
│  (left 73% only — excludes       │
│   mana cost icons on the right)  │
│                                  │
│  ... artwork ...                 │
│                                  │
└──────────────────────────────────┘
```

The proportions (`_NAME_TOP = 0.034`, `_NAME_BOTTOM = 0.115`, `_NAME_RIGHT = 0.730`)
are based on the standard MTG card layout used since ~8th Edition. They work for most
frame styles (classic, M15, post-M15, showcase) but may need adjustment for very unusual
alternate frames.

---

## Step 3 — Image Enhancement

**File:** `preprocessing/image_utils.py` → `enhance_for_ocr()`

The name crop is then enhanced before being passed to EasyOCR:

### Grayscale conversion

EasyOCR's CRNN recognition model operates on grayscale internally. Converting upfront
removes any colour bias and reduces the data the model needs to process.

### CLAHE — Contrast Limited Adaptive Histogram Equalization

Card photos often have uneven lighting (one corner brighter than another, shadows, flash glare).
Plain histogram equalization stretches contrast globally, which can make well-lit areas
look washed out while fixing dark areas.

**CLAHE** divides the image into small tiles (`tileGridSize=(4, 4)`) and equalizes each
independently, then blends the results. `clipLimit=2.0` caps the contrast enhancement to
prevent noise in flat regions from being amplified.

Result: text that was hard to see due to uneven lighting becomes more legible.

> **Further reading:** [Histogram Equalization & CLAHE](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html)

### Fast Non-Local Means Denoising

JPEG compression and camera sensor noise appear as random pixel variation (grain).
Neural networks can misread these as text strokes. `cv2.fastNlMeansDenoising` removes
this grain by averaging each pixel against similar patches nearby.

`h=10` is the filter strength — higher values remove more noise but also blur fine detail.
At `h=10` we get noise reduction without softening the character edges too much.

> **Further reading:** [Non-Local Means Denoising](https://docs.opencv.org/4.x/d5/d69/tutorial_py_non_local_means.html)

---

## Step 4 — EasyOCR Recognition

**File:** `engines/easyocr_engine.py`

```python
reader = easyocr.Reader(["en"], gpu=False)
results = reader.readtext(image)
```

### `["en"]` — English only

Loading only the English model keeps memory usage down and avoids the recognition model
trying to interpret MTG's card name text as another language.

### `gpu=False`

Forces CPU inference. On Apple Silicon Macs, PyTorch can use the Metal Performance
Shaders (MPS) backend instead, but this currently triggers a warning
(`pin_memory not supported on MPS`). For a POC, CPU is fine — inference on the small
name crop takes ~1–2 seconds. A future optimisation would be to detect MPS availability
and suppress the warning or pass `gpu=True` to let EasyOCR use it.

### Reading the results

`readtext()` returns a list of `(bounding_box, text, confidence)` tuples — one per detected
text region. We join all fragments with a space in case the name is split into two bounding
boxes (e.g., "Lightning" and "Bolt" detected separately), then average their confidence scores.

---

## Step 5 — Text Normalisation

**File:** `preprocessing/image_utils.py` → `normalize_ocr_text()`

Even after good preprocessing, OCR engines make systematic character-level mistakes on
MTG card fonts. We apply a substitution table of known confusions:

| Raw OCR output | Corrected | Why it happens |
|---|---|---|
| `;` | `,` | The comma in names like "Auntie Ool, Cursewretch" has a visual tail that OCR reads as a semicolon |
| `\|` | `I` | A pipe character is visually identical to a capital I in many fonts |
| `` ` `` | `'` | Backtick confused for apostrophe in possessives ("Glen Elendra`s Answer") |
| Multiple spaces | Single space | Fragment joins occasionally produce double spaces |

This table will grow as more card images reveal new systematic errors.

---

## Accuracy Scoring

**File:** `benchmark/results.py`

When a `--ground-truth` name is provided, we score each engine's output using
**fuzzy string similarity** via [RapidFuzz](https://github.com/maxbachmann/RapidFuzz):

```python
similarity = fuzz.ratio(ground_truth, detected, processor=str.lower) / 100.0
```

`fuzz.ratio` computes a normalised Levenshtein similarity (0.0 = completely different,
1.0 = identical). It counts the minimum number of single-character edits (insert, delete,
replace) needed to turn one string into the other, then normalises by string length.

Example:
- Ground truth: `"Lightning Bolt"` (14 chars)
- Detected: `"Lightning Bo|t"` (14 chars, 1 substitution)
- Edit distance: 1
- Similarity: `(2 × (14 - 1)) / (14 + 14)` ≈ 0.929

> **Further reading:** [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance) ·
> [RapidFuzz docs](https://rapidfuzz.github.io/RapidFuzz/)

---

## Known Limitations

| Issue | Root cause | Status |
|---|---|---|
| Low-contrast backgrounds (dark table) | Canny can't find card edges | Strategy A now retries with lower thresholds |
| Borderless/extended-art cards | No rectangular card border to detect | Strategy B (type-line anchor) added as fallback |
| Foil/holographic cards | Reflections create false edges and text distortion | Not yet addressed |
| Rotated cards (landscape orientation) | Aspect ratio validator accepts both orientations, but name crop proportions assume portrait | Needs orientation detection |
| Stylised/alternate frame names (e.g. retro frame) | Name bar position may differ slightly | Crop constants may need adjustment |

---

## Tuning Levers

Things to experiment with if accuracy needs improving:

### Preprocessing

| Parameter | Location | Try |
|---|---|---|
| `clipLimit` in CLAHE | `enhance_for_ocr()` | Higher (3.0–4.0) for very low contrast images |
| `h` in denoising | `enhance_for_ocr()` | Lower (5–7) to preserve more edge detail |
| Upscale before OCR | — | `cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)` before passing to EasyOCR |
| Invert image | — | If name text is light-on-dark, `cv2.bitwise_not(image)` before OCR |

### EasyOCR `readtext()` parameters

| Parameter | Default | What it does |
|---|---|---|
| `detail` | `1` | `0` returns text strings only (no bboxes/confidence); `1` returns full tuples |
| `paragraph` | `False` | `True` merges nearby detections into paragraph blocks |
| `text_threshold` | `0.7` | Confidence threshold to accept a text region |
| `low_text` | `0.4` | Controls how aggressively CRAFT finds low-confidence text regions |
| `contrast_ths` | `0.1` | Below this contrast ratio, EasyOCR auto-adjusts the image |
| `adjust_contrast` | `0.5` | Target contrast for auto-adjustment |

Lowering `text_threshold` and `low_text` can help on faint or stylised text but increases
false positives.

> **Further reading:** [EasyOCR readtext() API reference](https://github.com/JaidedAI/EasyOCR#api-documentation)

---

## Further Reading

### OCR fundamentals
- [How OCR works — overview (Nanonets blog)](https://nanonets.com/blog/ocr-with-deep-learning/)
- [CRNN: An End-to-End Trainable Neural Network for Image-based Sequence Recognition (paper)](https://arxiv.org/abs/1507.05717)
- [CRAFT: Character-Region Awareness For Text Detection (paper)](https://arxiv.org/abs/1904.01941)

### OpenCV image processing
- [OpenCV Python tutorials (official)](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Image thresholding](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)
- [Morphological transformations (dilate, erode, etc.)](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)
- [Geometric transformations (perspective warp)](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html)

### String similarity / benchmarking
- [Levenshtein distance (Wikipedia)](https://en.wikipedia.org/wiki/Levenshtein_distance)
- [RapidFuzz library](https://rapidfuzz.github.io/RapidFuzz/)

### MTG card anatomy
- [MTG card layout reference (Scryfall)](https://scryfall.com/docs/syntax) — useful for understanding the fields we're working with
- Card dimensions: 63 mm × 88 mm (2.5 in × 3.5 in), same as a standard poker card
