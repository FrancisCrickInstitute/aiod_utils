# AI OnDemand (AIoD) Utilities

A central package to unify helpful utilities for AI OnDemand that are useful/used across the Nextflow pipeline, [Segment-Flow](https://github.com/FrancisCrickInstitute/Segment-Flow), and the [Napari plugin](https://github.com/FrancisCrickInstitute/aiod_napari). This primarily covers a centralisation of I/O and the implementation of RLE format.


## Installation
Requires Python 3.11 or 3.12.

Using pip:

```
pip install aiod_utils
```

Using `uv`:

```
uv add aiod_utils
```

For Bio-Formats support (e.g. `.lif`, `.czi`), install the optional extra:

```
pip install "aiod_utils[bioformats]"
```

## What's included

- **`aiod_utils.io`** — Load images via [BioIO](https://github.com/bioio-devs/bioio), with automatic reader selection for common formats (TIFF, OME-TIFF, Zarr, ND2, and more).
- **`aiod_utils.rle`** — Encode and decode segmentation masks (binary and instance) as COCO-compatible _Run-Length Encoding_, with save/load support.
- **`aiod_utils.stacks`** — Utilities for splitting large volumetric images into memory-bounded substacks for use in our Nextflow pipeline ([Segment-Flow](https://github.com/FrancisCrickInstitute/Segment-Flow)).
- **`aiod_utils.preprocess`** — Modular image preprocessing steps (e.g. CLAHE, downsampling) with a base class for defining custom steps. Easily extendable for use in [Segment-Flow](https://github.com/FrancisCrickInstitute/Segment-Flow) or our [Napari plugin](https://github.com/FrancisCrickInstitute/aiod_napari).


## Documentation

For the wider AIoD documentation, please see our [docs](https://franciscrickinstitute.github.io/aiod_docs/).

## License

MIT — see [LICENSE](LICENSE).