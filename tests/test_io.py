import builtins
import importlib
import sys
import warnings

import pytest

import aiod_utils.io as io_mod
from aiod_utils.io import _guess_reader, get_extension

# ── helpers ───────────────────────────────────────────────────────────────────


@pytest.fixture
def block_module(monkeypatch):
    """Make `import <name>` (and submodules) raise ModuleNotFoundError,
    regardless of whether the package is actually installed in this env."""

    def _block(name):
        real_import = builtins.__import__

        def fake_import(module_name, *args, **kwargs):
            if module_name == name or module_name.startswith(f"{name}."):
                raise ModuleNotFoundError(
                    f"No module named {module_name!r}", name=module_name
                )
            return real_import(module_name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        monkeypatch.delitem(sys.modules, name, raising=False)

    return _block


@pytest.fixture
def with_extensions(monkeypatch):
    """Pretend a further set of extensions is claimed by an installed plugin,
    as e.g. the optional bioformats extra would add."""

    def _with(*extensions):
        known = io_mod._known_extensions() + extensions
        monkeypatch.setattr(io_mod, "_known_extensions", lambda: known)

    return _with


# ── get_extension ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "fpath, expected",
    [
        ("img.tif", ".tif"),
        ("img.ome.tiff", ".ome.tiff"),
        # Compound extension wins over the shorter one it ends with
        ("img.OME.TIFF", ".ome.tiff"),
        # bioio-ome-zarr registers only ".zarr", bioio-imageio only ".jpg"
        ("img.ome.zarr", ".ome.zarr"),
        ("img.jpeg", ".jpeg"),
        # Dots in the stem are not extensions
        ("tile_x-2.5_y-1.0_B.ome.tiff", ".ome.tiff"),
        ("img_v1.2.3.tif", ".tif"),
        ("/data/exp.1/img.ome.zarr", ".ome.zarr"),
        # Unclaimed by any installed reader
        ("img.vsi", None),
        ("img.xyz", None),
        ("img", None),
    ],
)
def test_get_extension(fpath, expected):
    assert get_extension(fpath) == expected


def test_get_extension_finds_extension_from_installed_plugins(with_extensions):
    assert get_extension("img.vsi") is None
    with_extensions(".vsi")
    assert get_extension("img.vsi") == ".vsi"


# ── known extensions → dedicated lightweight readers ─────────────────────────


@pytest.mark.parametrize(
    "fname, module_name",
    [
        ("img.ome.tiff", "bioio_ome_tiff"),
        ("img.ome.tif", "bioio_ome_tiff"),
        ("img.tif", "bioio_tifffile"),
        ("img.tiff", "bioio_tifffile"),
        ("img.zarr", "bioio_ome_zarr"),
        ("img.ome.zarr", "bioio_ome_zarr"),
        ("img.jpg", "bioio_imageio"),
        ("img.jpeg", "bioio_imageio"),
        ("img.png", "bioio_imageio"),
        ("img.nd2", "bioio_nd2"),
        ("img.czi", "bioio_czi"),
        ("img.lif", "bioio_lif"),
    ],
)
def test_guess_reader_known_extensions(fname, module_name):
    expected = importlib.import_module(module_name).Reader
    assert _guess_reader(fname) is expected


@pytest.mark.parametrize(
    "fname, module_name",
    [
        ("tile_x-2.5_y-1.0_B.ome.tiff", "bioio_ome_tiff"),
        ("img_v1.2.3.tif", "bioio_tifffile"),
        ("scan.2024.10.01.czi", "bioio_czi"),
        ("plate_1.5x.ome.zarr", "bioio_ome_zarr"),
    ],
)
def test_guess_reader_ignores_dots_in_stem(fname, module_name):
    expected = importlib.import_module(module_name).Reader
    assert _guess_reader(fname) is expected


def test_guess_reader_warns_when_preferred_plugin_missing(block_module):
    block_module("bioio_tifffile")
    with pytest.warns(UserWarning, match="bioio_tifffile"):
        assert _guess_reader("img.tif") is None


# ── extensions claimed by a plugin we have no preference between ─────────────


@pytest.mark.parametrize("fname", ["img.bmp", "img.gif", "img.lsm", "img.jp2"])
def test_guess_reader_defers_to_bioio_for_unpreferred_extensions(fname):
    # An installed plugin handles these, so bioio resolves them itself rather
    # than us second-guessing which reader wins
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _guess_reader(fname) is None


def test_guess_reader_defers_for_extensions_only_bioformats_claims(with_extensions):
    # With the bioformats extra installed, its formats are claimed extensions -
    # bioio picks it as the sole candidate, so we neither import nor name it
    with_extensions(".vsi")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _guess_reader("img.vsi") is None


# ── extensions no installed reader claims ────────────────────────────────────


def test_guess_reader_warns_for_unclaimed_extension():
    with pytest.warns(UserWarning, match="bioformats2raw") as record:
        assert _guess_reader("img.vsi") is None
    message = str(record[0].message)
    assert "aiod_utils[bioformats]" in message
    # Report the extension we were given, not a matched one
    assert "'.vsi'" in message
