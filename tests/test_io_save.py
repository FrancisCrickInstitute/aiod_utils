from unittest.mock import patch

import dask.array as da
import numpy as np
import pytest
from aiod_utils.io import (
    _save_image_ome_zarr,
    save_image,
)

# --- Dispatch tests ---


@pytest.mark.parametrize(
    "ext, expected_helper",
    [
        (".tiff", "aiod_utils.io._save_image_ome_tiff"),
        (".ome.tiff", "aiod_utils.io._save_image_ome_tiff"),
        (".tif", "aiod_utils.io._save_image_ome_tiff"),
        (".ome.zarr", "aiod_utils.io._save_image_ome_zarr"),
        (".zarr", "aiod_utils.io._save_image_ome_zarr"),
        (".png", "aiod_utils.io._save_image_imageio"),
        (".jpg", "aiod_utils.io._save_image_imageio"),
        (".jpeg", "aiod_utils.io._save_image_imageio"),
    ],
)
def test_save_image_dispatches_by_extension(tmp_path, ext, expected_helper):
    data = np.zeros((4, 4), dtype=np.uint8)
    fpath = str(tmp_path / f"test{ext}")
    with patch(expected_helper) as mock_helper:
        save_image(data, fpath)
    mock_helper.assert_called_once_with(data, fpath)


# --- Unsupported extension ---


@pytest.mark.parametrize("ext", [".nd2", ".xyz", ".bmp"])
def test_save_image_raises_for_unsupported_extension(tmp_path, ext):
    data = np.zeros((4, 4), dtype=np.uint8)
    fpath = str(tmp_path / f"test{ext}")
    with pytest.raises(ValueError, match="Unsupported extension"):
        save_image(data, fpath)


# --- OME-Zarr roundtrip ---


def test_save_image_ome_zarr_roundtrip(tmp_path):
    from bioio import BioImage

    data = np.random.randint(0, 255, (1, 1, 4, 8, 8), dtype=np.uint8)
    fpath = str(tmp_path / "test.ome.zarr")
    save_image(data, fpath)

    img = BioImage(fpath)
    loaded = img.get_image_data("TCZYX")
    np.testing.assert_array_equal(data, loaded)


# --- OME-TIFF roundtrip ---


def test_save_image_ome_tiff_roundtrip(tmp_path):
    from bioio import BioImage

    data = np.random.randint(0, 255, (1, 1, 4, 8, 8), dtype=np.uint8)
    fpath = str(tmp_path / "test.ome.tiff")
    save_image(data, fpath)

    img = BioImage(fpath)
    loaded = img.get_image_data("TCZYX")
    np.testing.assert_array_equal(data, loaded)


# --- PNG roundtrip ---


def test_save_image_png_roundtrip(tmp_path):
    from skimage.io import imread

    data = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
    fpath = str(tmp_path / "test.png")
    save_image(data, fpath)

    loaded = imread(fpath)
    np.testing.assert_array_equal(data, loaded)


# --- RGB roundtrips ---


def test_save_image_png_rgb_roundtrip(tmp_path):
    from skimage.io import imread

    data = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    fpath = str(tmp_path / "test_rgb.png")
    save_image(data, fpath)

    loaded = imread(fpath)
    np.testing.assert_array_equal(data, loaded)


def test_save_image_jpeg_rgb_roundtrip(tmp_path):
    # JPEG is lossy so only check shape is preserved
    from skimage.io import imread

    data = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    fpath = str(tmp_path / "test_rgb.jpg")
    save_image(data, fpath)

    loaded = imread(fpath)
    assert loaded.shape == data.shape


def test_save_image_tiff_rgb_roundtrip(tmp_path):
    # Save an RGB image as PNG first so bioio loads it with the S (samples) dimension,
    # then save to TIFF via the BioImage path and verify pixel data is preserved.
    from skimage.io import imsave

    from aiod_utils.io import load_image, load_image_data

    data = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    png_path = tmp_path / "source_rgb.png"
    imsave(str(png_path), data)

    img = load_image(png_path)
    tiff_path = str(tmp_path / "test_rgb.tiff")
    save_image(img, tiff_path)

    loaded = load_image_data(tiff_path, dim_order="YXC", rgb_as_channels=True, expand_dims=False)
    np.testing.assert_array_equal(data, loaded)


# --- AttributeError fallback via skimage ---


def test_save_image_fallback_on_missing_writer(tmp_path):
    from pathlib import Path

    data = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
    fpath = str(tmp_path / "test.tiff")

    with patch(
        "aiod_utils.io._save_image_ome_tiff",
        side_effect=AttributeError("OmeTiffWriter"),
    ):
        save_image(data, fpath)

    assert Path(fpath).exists()
    from skimage.io import imread

    loaded = imread(fpath)
    np.testing.assert_array_equal(data, loaded)


# --- _save_image_ome_zarr with dask array ---


def test_save_image_ome_zarr_dask_array(tmp_path):
    from bioio import BioImage

    np_data = np.random.randint(0, 255, (1, 1, 4, 8, 8), dtype=np.uint8)
    dask_data = da.from_array(np_data, chunks=(1, 1, 2, 4, 4))
    fpath = str(tmp_path / "test_dask.ome.zarr")

    _save_image_ome_zarr(dask_data, fpath)

    img = BioImage(fpath)
    loaded = img.get_image_data("TCZYX")
    np.testing.assert_array_equal(np_data, loaded)
