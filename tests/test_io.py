import builtins
import importlib
import sys
import types

import pytest

from aiod_utils.io import _guess_reader

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
def fake_module(monkeypatch):
    """Register a fake module in sys.modules so `from <name> import Reader`
    succeeds without the real (heavy) package being installed."""

    def _fake(name, reader):
        module = types.ModuleType(name)
        module.Reader = reader
        monkeypatch.setitem(sys.modules, name, module)
        return module

    return _fake


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


# ── long tail: no dedicated reader, falls back to bioio-bioformats ──────────


def test_guess_reader_long_tail_uses_bioformats_when_available(fake_module):
    module = fake_module("bioio_bioformats", reader=object())
    assert _guess_reader("img.vsi") is module.Reader


def test_guess_reader_long_tail_warns_without_bioformats(block_module):
    block_module("bioio_bioformats")
    with pytest.warns(UserWarning, match="bioformats2raw") as record:
        result = _guess_reader("img.vsi")
    assert result is None
    assert "aiod_utils[bioformats]" in str(record[0].message)
