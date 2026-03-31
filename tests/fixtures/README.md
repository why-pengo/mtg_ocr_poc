# Test Image Fixtures

Place card images here for integration testing and benchmarking.

## Format

Alongside each image, there should be a matching `.txt` file containing the exact card name:

```
tests/fixtures/lightning_bolt.jpg
tests/fixtures/lightning_bolt.txt   ← contents: "Lightning Bolt"
```

## Running benchmarks against fixtures

```bash
.venv/bin/python main.py tests/fixtures/lightning_bolt.jpg --ground-truth "Lightning Bolt"
```

## Notes

- Image files are gitignored (`.jpg`, `.jpeg`, `.png`, `.webp`).
- Commit only the `.txt` ground-truth files.
- Aim for a mix of clean scans and real-world photos to get a fair benchmark.
