# DJAI external datasets

DJAI keeps external annotations separate from project source and never bundles copyrighted music.

## EDM-CUE

EDM-CUE accompanies *Cue Point Estimation using Object Detection* (Argüello, Lanzendörfer, Wattenhofer, ISMIR 2024). The paper reports 4,710 EDM tracks, 21,461 expert cue annotations, tempo/beat-grid metadata, and Deezer references. The public CUE-DETR checkpoint is `disco-eth/cue-detr`.

1. Obtain the annotation export from the authors' ISMIR project page.
2. Import CSV, JSON, or JSONL into DJAI:

```bash
djai edm-cue-import /path/to/downloaded/annotations.json -o datasets/edm-cue/annotations.jsonl
```

3. Inspect the normalized dataset:

```bash
djai edm-cue-stats datasets/edm-cue/annotations.jsonl
```

4. Optionally install the published cue model and predict cues on music you are authorized to process:

```bash
pip install -e ".[cue]"
DJAI_CUE_MODEL_ALLOW_DOWNLOAD=1 djai cue-predict audio/track.wav
```

For repeatable/offline use, download/cache the model once and set `DJAI_CUE_MODEL_ALLOW_DOWNLOAD=0` afterwards.

The importer preserves only annotation metadata. Source audio must come from music you own, license, or are otherwise authorized to process.
