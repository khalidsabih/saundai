# Rendered-audio measurement

The canonical Phase 1 design and configuration are documented in [QUALITY.md](QUALITY.md).

The short contract is:

```text
rank analytically → render top K → measure actual audio → choose a passing winner
```

The directed graph remains analytical. Measurement runs only for final pair or set-junction selection, and all candidate results are appended to local JSONL for future learning.
