import warnings
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
from bioio import BioImage, writers
from bioio.plugins import get_plugins
from bioio_base.exceptions import InvalidDimensionOrderingError
from bioio_base.reader import Reader

PathLike = str | Path
ImageLike = BioImage | np.ndarray | da.Array

# Separates the (image, preprocessing) prefix from the run hash in mask names
MASK_SEPARATOR = "_masks_"
# Marks the mask combined across every substack, as opposed to one substack
COMBINED_MASK_SUFFIX = "_all"


def _guess_reader(fpath: PathLike) -> type[Reader] | None:
    ext = "".join(Path(fpath).suffixes).lower()
    try:
        if ext.endswith((".ome.tiff", ".ome.tif")):
            from bioio_ome_tiff import Reader as OMETiffReader

            return OMETiffReader
        elif ext.endswith((".tif", ".tiff")):
            from bioio_tifffile import Reader as TiffReader

            return TiffReader
        elif ext.endswith((".zarr", ".ome.zarr")):
            from bioio_ome_zarr import Reader as ZarrReader

            return ZarrReader
        elif ext.endswith((".jpg", ".jpeg", ".png")):
            from bioio_imageio import Reader as ImageIOReader

            return ImageIOReader
        elif ext.endswith((".nd2",)):
            from bioio_nd2 import Reader as ND2Reader

            return ND2Reader
        elif ext in [".czi"]:
            from bioio_czi import Reader as CZIReader

            return CZIReader
        elif ext in [".lif"]:
            from bioio_lif import Reader as LIFReader

            return LIFReader
        else:
            # Long tail of formats with no dedicated lightweight reader: fall
            # back to bioio-bioformats if the (optional, heavy) extra is
            # installed, otherwise let the except block below point the user
            # at it, or at pre-converting with bioformats2raw.
            from bioio_bioformats import Reader as BioformatsReader

            return BioformatsReader
    except ModuleNotFoundError as e:
        message = (
            f"Recommended reader plugin {e.name} for file extension {ext} not installed"
        )
        if e.name == "bioio_bioformats":
            message += (
                ". Install aiod_utils[bioformats], or convert the file first "
                "with bioformats2raw and point at the converted output."
            )
        warnings.warn(message, stacklevel=2)
    return None


@dataclass(frozen=True)
class ImageId:
    """
    Identity of a source image, split into the parts consumers need.

    ``stem`` is for anything a user reads (e.g. napari layer names), ``value``
    is for anything that has to be unique on disk or across processes (mask
    filenames, the ``image_id`` CSV column Segment-Flow carries through).

    ``value`` always folds the extension in, even when nothing would collide.
    This ensures that image_ids always resolve to a unique result, regardless
    of context (i.e. other files that have the same name with diff ext).
    """

    stem: str
    ext: str

    @property
    def value(self) -> str:
        return f"{self.stem}_{self.ext.lstrip('.').replace('.', '_')}"

    def __str__(self) -> str:
        return self.value


def get_image_id(img_path: str | Path) -> ImageId:
    """
    Get a consistent identity for an image path to use as basis for derived paths
    (e.g. mask filenames)

    Strip accepted extensions (potentially multi-dot) by polling bioio for those extensions

    Centralised function here gives a better source of truth across e.g. Napari & Nextflow
    for expected filenames
    """
    name = Path(img_path).name
    if not Path(img_path).suffix:
        raise ValueError(
            f"Image path {img_path} has no extension, which is not supported!"
        )
    name_lower = name.lower()

    # get_plugins() returns extensions from all installed bioio reader plugins
    extension_mapping = get_plugins(use_cache=True)
    candidates = set(extension_mapping)
    # bioio-ome-zarr reader only lists .zarr as supported (not .ome.zarr!)
    # So add .ome.zarr manually (for now)
    candidates.add(".ome.zarr")

    # Match the longest recognized extension against the filename's end
    # We match bioio and match e.g. .ome.tiff first over .tiff
    for ext in sorted(candidates, key=len, reverse=True):
        if name_lower.endswith(ext):
            return ImageId(stem=name[: -len(ext)], ext=name[-len(ext) :])
    # If still no match, then raise an error so we can avoid more obscure errors later in the pipeline
    raise ValueError(
        f"Image path {img_path} has an unrecognized extension "
        f"'{Path(img_path).suffix}' - no installed bioio reader supports it. "
        f"Accepted extensions: {sorted(candidates)}"
    )


def validate_image_ids(img_paths: Sequence[PathLike]) -> list[ImageId]:
    """
    Get the ImageId for a batch of paths, erroring if any of them collide.
    A collision is where a filename and extension is shared, i.e. same name
    in diff directories. This would later collide in Nextflow work dirs.
    """
    paths = [Path(p) for p in img_paths]
    image_ids = [get_image_id(p) for p in paths]
    conflicts = defaultdict(list)
    for path, image_id in zip(paths, image_ids, strict=True):
        conflicts[image_id.value].append(path)
    colliding = {k: v for k, v in conflicts.items() if len(v) > 1}
    if colliding:
        detail = "\n".join(
            f"  {image_id}: {[str(p) for p in ps]}"
            for image_id, ps in colliding.items()
        )
        raise ValueError(
            "Cannot derive unique image_id for the following image(s) - they "
            "share both a filename and extension:\n"
            f"{detail}\n"
            "Deduplicate the input, or rename/move one of each conflicting set "
            "of files, before rerunning."
        )
    return image_ids


def get_mask_prefix(
    image_id: ImageId | str,
    prep_hash: str | None = None,
) -> str:
    """
    Utility to get mask prefix from identity and preprocessing hash
    Avoids having to parse filenames when prefix is needed!

    ``image_id`` is normally an ImageId (or its ``value``, e.g. read back from
    Segment-Flow's image_id column). A display form of the identity is also
    accepted, so callers building a human-readable name (like aiod_napari's
    mask layer names) share this, the result then follows the identity given,
    not necessarily a real filename.
    """
    prep_suffix = f"_{prep_hash}" if prep_hash else ""
    return f"{image_id}{prep_suffix}"


def get_mask_name(
    run_hash: str,
    image_id: ImageId | str | None = None,
    image_path: str | Path | None = None,
    prep_hash: str | None = None,
) -> str:
    """
    Canonical mask filename stem, so that any consumer needing to predict a
    Segment-Flow mask filename before it exists (e.g. aiod_napari's file
    watcher) can call this.

    Uses ImageId (or an already-stringified ImageId.value), not a
    bare stem, as the extension is part of what makes the filename unique.

    NOTE: Segment-Flow's equivalent (getMaskName in main.nf) has to compute
    this independently. Nextflow's process `output:` declarations need the
    filename pattern known before the script runs, which a Python func can't
    provide. If this format ever changes, update both places!
    """
    if image_id is None:
        if image_path is None:
            raise ValueError("Either image_id or image_path must be provided")
        image_id = get_image_id(image_path)
    return f"{get_mask_prefix(image_id, prep_hash)}{MASK_SEPARATOR}{run_hash}"


def get_mask_prefix_from_name(fpath: PathLike) -> str:
    """
    Recover the (image, [preprocessing]) prefix from a mask filename.

    Mainly used by file watchers that need to handle filenames directly.
    """
    # NOTE: rsplit used in case the original filename had MASK_SEPARATOR in it
    return Path(fpath).stem.rsplit(MASK_SEPARATOR, 1)[0]


def get_combined_mask_name(mask_name: str, extension: str | None = None) -> str:
    """
    Name of the mask combined across every substack, optionally with extension

    NOTE: The `mask_name` is expected to be the formatted output from `get_mask_name()`
    """
    name = f"{mask_name}{COMBINED_MASK_SUFFIX}"
    return f"{name}.{extension}" if extension else name


def is_combined_mask(fpath: PathLike) -> bool:
    """Whether a mask file is the combined one rather than a single substack"""
    return Path(fpath).stem.endswith(COMBINED_MASK_SUFFIX)


def guess_rgba(img: BioImage):
    # https://github.com/bioio-devs/bioio/issues/174#issuecomment-3843003521
    return "S" in img.dims.order


def load_image_data(
    image: PathLike | BioImage,
    dim_order: str = "CZYX",
    as_dask: bool = False,
    rgb_as_channels=True,
    expand_dims=True,
    **kwargs,
) -> np.ndarray | da.Array:
    """
    Returns data array without any associated metadata.
    Replaces legacy flagged mode of load_image:
        load_image(...,) => load_image(...,)
        load_image(..., return_array=True) => load_image_data(...)
        load_image(..., return_dask=True) => load_image_data(..., as_dask=True)

    Inputs
    ======

    image: file path or BioImage object

    Note
    ====

    In Bioio, RGB images by default store the RGB dimension as in samples 'S', separate from channels 'C' (see bioio-devs/bioio#174). If `rgb_as_channels` is True, and if 'C' is requested in the output `dim_order`, the 'S' dimension will be remapped to 'C'.
    """
    if isinstance(image, (str, Path)):
        image = load_image(image, **kwargs)
    elif len(kwargs):
        warnings.warn(
            f"load_image_data() received unexpected kwargs {kwargs}, which will be ignored",
            stacklevel=2,
        )
    # Check the dim_order, and remap obvious aliases
    dim_order = dim_order.upper().translate(str.maketrans("DHW", "ZYX"))
    if (
        rgb_as_channels
        and "C" in dim_order
        and "S" not in dim_order
        and guess_rgba(image)
    ):
        if getattr(image.dims, "C", 1) > 1:
            raise NotImplementedError("Multi-channel RGB(A) images not supported")
        dim_order = dim_order.replace("C", "S")
    # Keep only actual dims if singleton expansion disabled
    if not expand_dims:
        dim_order = "".join(
            d for d in dim_order if d in image.standard_metadata.dimensions_present
        )
    return (
        image.get_image_dask_data(dimension_order_out=dim_order)
        if as_dask
        else image.get_image_data(dimension_order_out=dim_order)
    )


def load_image(
    fpath: PathLike,
    reader: type[Reader] | None = None,
) -> BioImage:
    # Load the image with the requested reader
    # If no reader provided, guess an appropriate plugin or fall back on bioio default plugin order
    fpath = Path(fpath)
    if not fpath.exists():
        raise FileNotFoundError(f"File {fpath} does not exist.")
    return BioImage(fpath, reader=reader or _guess_reader(fpath))


def save_image(
    data: ImageLike,
    fpath: PathLike,
    dim_order: str = "CZYX",
):
    ext = "".join(Path(fpath).suffixes).lower()
    if ext.endswith((".ome.tiff", ".ome.tif", ".tif", ".tiff")):
        try:
            _save_image_ome_tiff(data, fpath, dim_order)
        except AttributeError as exc:
            # Fall back on tifffile for missing writer
            from tifffile import imwrite

            data = (
                load_image_data(
                    data,
                    dim_order=dim_order,
                    as_dask=False,
                )
                if isinstance(data, (BioImage, Path, str))
                else data
            )
            # TODO AIOD-315: can't deal with this scenario until load_image_data returns dim string
            if len(dim_order) != len(data.shape):
                raise NotImplementedError(
                    "Cannot use tifffile to save image with unspecified dimensions"
                ) from exc
            imwrite(fpath, data, metadata={"axes": dim_order})
    elif ext.endswith((".zarr", ".ome.zarr")):
        try:
            _save_image_ome_zarr(data, fpath, dim_order)
        except AttributeError as exc:
            raise NotImplementedError(
                "Cannot save to zarr without bioio-zarr installed."
            ) from exc
    else:
        raise ValueError(f"Unsupported extension: {ext}")


def _save_image_ome_zarr(data: ImageLike, fpath: PathLike, dim_order="CZYX"):
    if not hasattr(writers, "OMEZarrWriter"):
        raise AttributeError("OMEZarrWriter")
    if isinstance(data, BioImage):
        data = load_image_data(data, dim_order=dim_order)
    # Ensure axes_names length matches data ndim; take trailing axes if dim_order is longer
    if len(dim_order) > data.ndim:
        # NOTE: revisit this for AIOD-315
        dim_order = dim_order[-data.ndim :]
    elif len(dim_order) < data.ndim:
        raise InvalidDimensionOrderingError(
            f"dim_order '{dim_order}' has fewer dims than data shape {data.shape}"
        )
    writers.OMEZarrWriter(
        store=str(fpath),
        level_shapes=data.shape,
        dtype=data.dtype,
        axes_names=[a.lower() for a in dim_order],
    ).write_full_volume(data)


def _save_image_ome_tiff(data: ImageLike, fpath: PathLike, dim_order="CZYX"):
    if not hasattr(writers, "OmeTiffWriter"):
        raise AttributeError("OmeTiffWriter")
    if isinstance(data, BioImage):
        data = load_image_data(data, dim_order=dim_order)
    writers.OmeTiffWriter.save(data, fpath, dim_order=dim_order)


def image_paths_to_csv(
    image_paths: Sequence[PathLike] | PathLike,
    output_csv_path: PathLike,
    dimensions: Sequence[dict[str, int]] | dict[str, int] | None = None,
    dtypes: Sequence[str | np.dtype] | str | np.dtype | None = None,
    overwrite: bool = False,
    **kwargs,
):
    """
    Write image shape details to a csv file, given an input image path or list of paths.
    Optionally provide shape details (per image path) to be written in the form of dimensions dict with keys from STCZYX:
        dimensions = {
            'X':...,
            'Z':...,
        }
    If shape info are not provided or are incomplete, will attempt to read from the image metadata if available.
    Will raise FileExistsError if overwrite=False and output_csv_path exists.
    Any additional kwargs will be forwarded to pandas.DataFram.to_csv()
    """

    if not overwrite and Path(output_csv_path).exists():
        raise FileExistsError(
            f"Output csv file {output_csv_path} already exists and overwrite is set to False."
        )

    output = defaultdict(list)

    if isinstance(image_paths, (str, Path)):
        image_paths = [image_paths]
    if dimensions:
        if isinstance(dimensions, dict):
            dimensions = [dimensions]
        if len(dimensions) != len(image_paths):
            raise ValueError(
                "If providing dimensions, must provide one dimensions dict per image path."
            )
    else:
        # Fetch dimensions for each image from metadata
        # TODO: When this is implemented, might as well enable fetching dtype while we're at it
        raise NotImplementedError(
            "Fetching dimensions from image metadata not yet implemented."
        )
    if dtypes is not None:
        if isinstance(dtypes, (str, np.dtype)):
            dtypes = [dtypes]
        if len(dtypes) != len(image_paths):
            raise ValueError(
                "If providing dtypes, must provide one dtype per image path."
            )
    dt_iter = dtypes if dtypes is not None else [None] * len(image_paths)
    for path, shape, dt in zip(image_paths, dimensions, dt_iter, strict=True):
        output["img_path"].append(str(path))
        try:
            output["num_slices"].append(shape.get("Z", 1))
            output["height"].append(shape.get("Y") or shape["H"])  # raises KeyError
            output["width"].append(shape.get("X") or shape["W"])  # raises KeyError
            output["channels"].append(shape.get("C", 1))
        except KeyError as e:
            # NOTE: this message will give keyerror for H or W, without hinting to use Y and X instead
            raise ValueError(
                f"Dimensions dict for image {path} is missing required key: {e}"
            ) from e
        if dt is not None:
            output["dtype"].append(np.dtype(dt).name)
        else:
            output["dtype"].append(None)
    df = pd.DataFrame(output)
    df.to_csv(output_csv_path, **kwargs)


def extract_idxs_from_fname(fname: str, downsample_factor: Sequence[int] | None = None):
    # Extract the indices from the filename
    idx_ranges = Path(fname).stem.split("_")[-3:]
    start_x, end_x = map(int, idx_ranges[0].split("x")[1].split("-"))
    start_y, end_y = map(int, idx_ranges[1].split("y")[1].split("-"))
    start_z, end_z = map(int, idx_ranges[2].split("z")[1].split("-"))
    # Apply downsampling to indices if provided
    if downsample_factor is not None:
        if len(downsample_factor) == 2:
            down_y, down_x = downsample_factor
            down_z = 1
        else:
            down_z, down_y, down_x = downsample_factor
        start_x, end_x = round_idxs(start_x, end_x, down_x)
        start_y, end_y = round_idxs(start_y, end_y, down_y)
        start_z, end_z = round_idxs(start_z, end_z, down_z)
    return start_x, end_x, start_y, end_y, start_z, end_z


def round_idxs(start: int, end: int, downsample_factor: int):
    """
    When splitting and downsampling, we end up with indivisible block sizes.
    We handle this by removing whatever is getting padded in block_reduce
    We need to convert idxs in fnames to downsampled idxs, which may not divide cleanly.
    If we round down the start we increase size, so need to round different according to whether start or end is indivisable.
    """
    if (end - start) % downsample_factor != 0:
        start = int(np.ceil(start / downsample_factor))
        end = int(np.floor(end / downsample_factor))
    else:
        start = int(np.floor(start / downsample_factor))
        end = int(np.floor(end / downsample_factor))
    return start, end


def check_dtype(arr: np.ndarray, max_val: int | None = None):
    # Get the max value in the array if not provided
    if max_val is None:
        max_val = arr.max()
    # Get the appropriate dtype from the max value
    if max_val <= np.iinfo(np.uint8).max:
        best_dtype = np.uint8
    elif max_val <= np.iinfo(np.uint16).max:
        best_dtype = np.uint16
    # Surely it doesn't need more than 32 bits...
    else:
        best_dtype = np.uint32
    return best_dtype


def reduce_dtype(arr: np.ndarray, max_val: int | None = None):
    # Get the lowest bit dtype for the array
    best_dtype = check_dtype(arr, max_val)
    # If the current dtype is already the best, return the array
    if arr.dtype == best_dtype:
        return arr
    # Otherwise convert it
    else:
        return arr.astype(best_dtype, copy=False)
