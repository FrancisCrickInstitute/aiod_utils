from pathlib import Path
from unittest.mock import patch

import dask.array as da
import numpy as np
import pytest

from aiod_utils.io import _save_image_ome_zarr, load_image, load_image_data, save_image

# --- Dispatch tests ---


@pytest.mark.parametrize(
    "ext, expected_helper",
    [
        (".tiff", "aiod_utils.io._save_image_ome_tiff"),
        (".ome.tiff", "aiod_utils.io._save_image_ome_tiff"),
        (".tif", "aiod_utils.io._save_image_ome_tiff"),
        (".ome.zarr", "aiod_utils.io._save_image_ome_zarr"),
        (".zarr", "aiod_utils.io._save_image_ome_zarr"),
    ],
)
def test_save_image_dispatches_by_extension(tmp_path, ext, expected_helper):
    data = np.zeros((4, 8), dtype=np.uint8)
    fpath = str(tmp_path / f"test{ext}")
    with patch(expected_helper) as mock_helper:
        save_image(data, fpath)
    if "ome_zarr" in expected_helper:
        mock_helper.assert_called_once_with(data, fpath, "CZYX")
    else:
        mock_helper.assert_called_once_with(data, fpath, "CZYX")


# --- Unsupported extension ---


@pytest.mark.parametrize("ext", [".nd2", ".xyz", ".bmp", ".png", ".jpg", ".jpeg"])
def test_save_image_raises_for_unsupported_extension(tmp_path, ext):
    data = np.zeros((4, 8), dtype=np.uint8)
    fpath = str(tmp_path / f"test{ext}")
    with pytest.raises(ValueError, match="Unsupported extension"):
        save_image(data, fpath)


# --- OME-Zarr roundtrip ---


def test_save_image_ome_zarr_roundtrip(tmp_path):
    from bioio import BioImage

    data = np.random.randint(0, 255, (1, 1, 4, 8, 16), dtype=np.uint8)
    fpath = str(tmp_path / "test.ome.zarr")
    save_image(data, fpath, dim_order="TCZYX")

    img = BioImage(fpath)
    loaded = img.get_image_data("TCZYX")
    np.testing.assert_array_equal(data, loaded)


# --- OME-TIFF roundtrip ---


def test_save_image_ome_tiff_roundtrip(tmp_path):
    from bioio import BioImage

    data = np.random.randint(0, 255, (1, 1, 4, 8, 16), dtype=np.uint8)
    fpath = str(tmp_path / "test.ome.tiff")
    save_image(data, fpath, "TCZYX")

    img = BioImage(fpath)
    loaded = img.get_image_data("TCZYX")
    np.testing.assert_array_equal(data, loaded)


# --- OME-Zarr roundtrip with CZYX dim_order (preprocessing pipeline scenario) ---


def test_save_load_ome_zarr_czyx_roundtrip(tmp_path):
    """Mimics preprocessing: save 4D CZYX array to OME-Zarr, reload with load_image_data."""
    from aiod_utils.io import load_image_data

    data = np.random.randint(0, 255, (1, 4, 32, 64), dtype=np.uint8)  # CZYX
    fpath = str(tmp_path / "preprocessed.ome.zarr")
    save_image(data, fpath, dim_order="CZYX")

    # Roundtrip: load as the model scripts do
    loaded = load_image_data(fpath, dim_order="CZYX")
    np.testing.assert_array_equal(data, loaded)


def test_save_load_ome_zarr_squeezed_roundtrip(tmp_path):
    """Mimics preprocessing with squeezed singleton dims: save 3D ZYX, reload as CZYX."""
    from aiod_utils.io import load_image_data

    # Simulate: original CZYX was (1, 4, 64, 64), C=1 squeezed → ZYX
    data_3d = np.random.randint(0, 255, (4, 32, 64), dtype=np.uint8)
    fpath = str(tmp_path / "squeezed.ome.zarr")
    save_image(data_3d, fpath, dim_order="ZYX")

    # load_image_data expands missing C as singleton
    loaded = load_image_data(fpath, dim_order="CZYX")
    assert loaded.shape == (1, 4, 32, 64)
    np.testing.assert_array_equal(data_3d, loaded.squeeze())


def test_save_load_ome_zarr_multichannel_roundtrip(tmp_path):
    """3-channel z-stack saved and reloaded as CZYX."""
    from aiod_utils.io import load_image_data

    data = np.random.randint(0, 255, (3, 8, 32, 64), dtype=np.uint8)  # CZYX
    fpath = str(tmp_path / "multi_ch.ome.zarr")
    save_image(data, fpath, dim_order="CZYX")

    loaded = load_image_data(fpath, dim_order="CZYX")
    np.testing.assert_array_equal(data, loaded)


# --- AttributeError fallback via tifffile ---


@pytest.mark.parametrize("dim_order", ["CZYX", "ZYX", "ZXY"])
def test_save_image_fallback_raises_on_mismatched_dims(tmp_path, dim_order):
    """Fallback to tifffile raises when dim_order doesn't match data ndim."""
    data = np.random.randint(0, 255, (32, 64), dtype=np.uint8)
    fpath = str(tmp_path / "test.tiff")

    with (
        patch(
            "aiod_utils.io._save_image_ome_tiff",
            side_effect=AttributeError("OmeTiffWriter"),
        ),
        pytest.raises(NotImplementedError),
    ):
        save_image(data, fpath, dim_order=dim_order)

    assert not Path(fpath).exists()


@pytest.mark.parametrize("dim_order", ["XY", "YX"])
def test_save_image_fallback_roundtrip(tmp_path, dim_order):
    """Fallback to tifffile succeeds and roundtrips when dims match."""
    data = np.random.randint(0, 255, (32, 64), dtype=np.uint8)
    fpath = str(tmp_path / "test.tiff")

    with patch(
        "aiod_utils.io._save_image_ome_tiff",
        side_effect=AttributeError("OmeTiffWriter"),
    ):
        save_image(data, fpath, dim_order=dim_order)

    assert Path(fpath).exists()
    from tifffile import imread

    loaded = imread(fpath)
    np.testing.assert_array_equal(data, loaded)

    loaded = load_image(fpath)
    np.testing.assert_array_equal(
        data, load_image_data(loaded, expand_dims=False, dim_order=dim_order)
    )


# --- _save_image_ome_zarr with dask array ---


def test_save_image_ome_zarr_dask_array(tmp_path):
    from bioio import BioImage

    np_data = np.random.randint(0, 255, (1, 1, 4, 8, 16), dtype=np.uint8)
    dask_data = da.from_array(np_data, chunks=(1, 1, 2, 4, 4))
    fpath = str(tmp_path / "test_dask.ome.zarr")

    _save_image_ome_zarr(dask_data, fpath, dim_order="TCZYX")

    img = BioImage(fpath)
    loaded = img.get_image_data("TCZYX")
    np.testing.assert_array_equal(np_data, loaded)
