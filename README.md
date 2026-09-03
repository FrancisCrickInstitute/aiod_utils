# AI OnDemand (AIoD) Utilities

A central package to unify helpful utilities for AI OnDemand that are useful/used across the Nextflow pipeline, [Segment-Flow](https://github.com/FrancisCrickInstitute/Segment-Flow), and the [Napari plugin](https://github.com/FrancisCrickInstitute/aiod_napari). This primarily covers a centralisation of I/O and the implementation of RLE format.


## Installation
Requires Python 3.11 or 3.12.

Using pip:

```bash
pip install aiod_utils
```

Using `uv`:

```bash
uv add aiod_utils  # or uv pip install aiod_utils
```

For Bio-Formats support, install the optional extra:

```bash
pip install "aiod_utils[bioformats]"
```

## What's included

- **`aiod_utils.io`** — Load images via [BioIO](https://github.com/bioio-devs/bioio), with automatic reader selection for common formats (TIFF, OME-TIFF, Zarr, ND2, and more). Also centralises image/mask naming (`get_image_id`, `get_mask_name`, `get_mask_prefix`, and friends) so the Napari front-end and Segment-Flow backend derive filenames identically.
- **`aiod_utils.rle`** — Encode and decode segmentation masks (binary and instance) as COCO-compatible _Run-Length Encoding_, with save/load support.
- **`aiod_utils.stacks`** — Utilities for splitting large volumetric images into memory-bounded substacks for use in our Nextflow pipeline ([Segment-Flow](https://github.com/FrancisCrickInstitute/Segment-Flow)).
- **`aiod_utils.preprocess`** — Modular image preprocessing steps (e.g. CLAHE, downsampling) with a base class for defining custom steps. Easily extendable for use in [Segment-Flow](https://github.com/FrancisCrickInstitute/Segment-Flow) or our [Napari plugin](https://github.com/FrancisCrickInstitute/aiod_napari). Includes `get_prep_hash`/`hash_params_str` for deriving a short, deterministic hash of a preprocessing config, shared by both ends for cache-consistent naming.


## Documentation

For the wider AIoD documentation, please see our [docs](https://franciscrickinstitute.github.io/aiod_docs/).

## License

MIT — see [LICENSE](LICENSE).