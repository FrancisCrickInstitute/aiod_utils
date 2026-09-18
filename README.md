# AI OnDemand (AIoD) Utilities

[![PyPI](https://img.shields.io/pypi/v/aiod-utils.svg)](https://pypi.org/project/aiod-utils/)
[![Python versions](https://img.shields.io/pypi/pyversions/aiod-utils.svg)](https://pypi.org/project/aiod-utils/)
[![Tests](https://github.com/FrancisCrickInstitute/aiod_utils/actions/workflows/test.yml/badge.svg)](https://github.com/FrancisCrickInstitute/aiod_utils/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-aiod__docs-1f6feb.svg)](https://franciscrickinstitute.github.io/aiod_docs/sections/utilities/)

A central package to unify helpful utilities for AI OnDemand that are useful/used across the Nextflow pipeline, [Segment-Flow](https://github.com/FrancisCrickInstitute/Segment-Flow), and the [Napari plugin](https://github.com/FrancisCrickInstitute/aiod_napari). This primarily covers a centralisation of I/O and the implementation of RLE format.

Nothing here is AIoD-specific at the API level. The image I/O, RLE, and substack helpers are usable on their own in any bioimage analysis project!

## Requirements

Python 3.11 or 3.12.

## Installation

Using `uv`:

```bash
uv add aiod_utils  # or uv pip install aiod_utils
```

Using pip:

```bash
pip install aiod_utils
```

For Bio-Formats support, install the optional extra:

```bash
uv add "aiod_utils[bioformats]"
```

## Quick start

Load an image with an automatically-selected reader, then round-trip a mask through the RLE format:

```python
import aiod_utils.io as io
import aiod_utils.rle as rle

img = io.load_image("my_image.ome.tiff")  # a BioIO BioImage
data = img.get_image_data("ZYX")

encoded = rle.encode(mask, mask_type="instance")  # mask: np.ndarray, 2D or 3D
rle.save_encoding(encoded, "mask.rle")

decoded, metadata = rle.decode(rle.load_encoding("mask.rle"))
```

## What's included

- **`aiod_utils.io`** — Load images via [BioIO](https://github.com/bioio-devs/bioio), with automatic reader selection for common formats (TIFF, OME-TIFF, Zarr, ND2, and more), and save them back out as OME-TIFF or OME-Zarr. Also centralises image/mask naming so the Napari front-end (and potential others) and [Segment-Flow](https://github.com/FrancisCrickInstitute/Segment-Flow) backend derive filenames identically.
- **`aiod_utils.rle`** — Encode and decode segmentation masks (binary and instance) as COCO-compatible _Run-Length Encoding_, with save/load support.
    - Note that there are some optimisations here to help improve encode/decode times for dense segmentation masks, particularly for storing instance masks!
- **`aiod_utils.stacks`** — Utilities for splitting large volumetric images into memory-bounded substacks for use in our Nextflow pipeline ([Segment-Flow](https://github.com/FrancisCrickInstitute/Segment-Flow)). Is generally useful for dividing images/arrays into subsets to parallelise/iterate over, with optional memory budget.
- **`aiod_utils.preprocess`** — Modular image preprocessing steps (e.g. CLAHE, downsampling) with a base class for defining custom steps. Easily extendable for use in [Segment-Flow](https://github.com/FrancisCrickInstitute/Segment-Flow) or our [Napari plugin](https://github.com/FrancisCrickInstitute/aiod_napari).

## Documentation

Full documentation for AIoD lives at **[franciscrickinstitute.github.io/aiod_docs](https://franciscrickinstitute.github.io/aiod_docs/)**.

| Topic | Link |
| --- | --- |
| What each module does, in depth | [Centralised Utilities](https://franciscrickinstitute.github.io/aiod_docs/sections/utilities/) |
| The RLE format and why it exists | [Customised RLE format](https://franciscrickinstitute.github.io/aiod_docs/sections/utilities/#customised-run-length-encoding-format) |
| How substacks are sized and used | [Segmenting at scale](https://franciscrickinstitute.github.io/aiod_docs/sections/concepts/#segmenting-at-scale) |
| Adding a new preprocessing step | [Expanding AIoD](https://franciscrickinstitute.github.io/aiod_docs/sections/contributing/expanding/#preprocessing-function) |
| How AIoD fits together | [AIoD Concepts](https://franciscrickinstitute.github.io/aiod_docs/sections/concepts/) |

## Contributing

Contributions are very welcome — see the [AIoD Developer Guide](https://franciscrickinstitute.github.io/aiod_docs/sections/contributing/developing/) for setting up a cross-repo development environment.

This package is a pinned dependency of every model conda environment in Segment-Flow as well as the Napari plugin, so changes here have a wide blast radius. Run the tests before opening a PR:

```bash
uv sync
uv run pytest
```

## Support

Found a bug or want to request a feature? Please [open an issue](https://github.com/FrancisCrickInstitute/aiod_utils/issues). For usage problems, the docs [Troubleshooting](https://franciscrickinstitute.github.io/aiod_docs/sections/support/troubleshooting/) and [Support](https://franciscrickinstitute.github.io/aiod_docs/sections/support/) pages are the best starting points.

## License

MIT — see [LICENSE](LICENSE).
