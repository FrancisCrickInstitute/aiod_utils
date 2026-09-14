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

- **`aiod_utils.io`** — Load images via [BioIO](https://github.com/bioio-devs/bioio), with automatic reader selection for common formats (TIFF, OME-TIFF, Zarr, ND2, and more), and save them back out as OME-TIFF or OME-Zarr. Also centralises image/mask naming so the Napari front-end (and potential others) and [Segment-Flow](https://github.com/FrancisCrickInstitute/Segment-Flow) backend derive filenames identically.
- **`aiod_utils.rle`** — Encode and decode segmentation masks (binary and instance) as COCO-compatible _Run-Length Encoding_, with save/load support.
    - Note that there are some optimisations here to help improve encode/decode times for dense segmentation masks, particularly for storing instance masks!
- **`aiod_utils.stacks`** — Utilities for splitting large volumetric images into memory-bounded substacks for use in our Nextflow pipeline ([Segment-Flow](https://github.com/FrancisCrickInstitute/Segment-Flow)). Is generally useful for dividing images/arrays into subsets to parallelise/iterate over, with optional memory budget.
- **`aiod_utils.preprocess`** — Modular image preprocessing steps (e.g. CLAHE, downsampling) with a base class for defining custom steps. Easily extendable for use in [Segment-Flow](https://github.com/FrancisCrickInstitute/Segment-Flow) or our [Napari plugin](https://github.com/FrancisCrickInstitute/aiod_napari).


## Documentation

For the wider AIoD documentation, please see our [docs](https://franciscrickinstitute.github.io/aiod_docs/).

## License

MIT — see [LICENSE](LICENSE).