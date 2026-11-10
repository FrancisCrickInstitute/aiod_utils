import builtins
import importlib
import os
import sys
import types

import pytest

from aiod_utils.io import (
    BIOFORMATS_VERSION,
    _configure_bioformats_jvm,
    _guess_reader,
    prefetch_bioformats,
)

# ── helpers ───────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_scyjava_config(monkeypatch):
    """Stand-in for scyjava.config that records calls, so JVM configuration can
    be checked without Java or the bioformats extra installed."""
    calls = {}
    config = types.ModuleType("scyjava.config")
    config.set_cache_dir = lambda cache_dir: calls.__setitem__("cache_dir", cache_dir)
    config.set_java_constraints = lambda **kwargs: calls.__setitem__(
        "constraints", kwargs
    )
    scyjava = types.ModuleType("scyjava")
    scyjava.config = config
    monkeypatch.setitem(sys.modules, "scyjava", scyjava)
    monkeypatch.setitem(sys.modules, "scyjava.config", config)
    return calls


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


def test_guess_reader_long_tail_uses_bioformats_when_available(
    fake_module, fake_scyjava_config, monkeypatch
):
    monkeypatch.delenv("BFF_JAVA_FETCH", raising=False)
    module = fake_module("bioio_bioformats", reader=object())
    assert _guess_reader("img.vsi") is module.Reader
    # The JVM starts lazily on first read, so configuring here is in time
    assert fake_scyjava_config["constraints"] == {"fetch": "auto"}


def test_guess_reader_long_tail_warns_without_bioformats(block_module):
    block_module("bioio_bioformats")
    with pytest.warns(UserWarning, match="bioformats2raw") as record:
        result = _guess_reader("img.vsi")
    assert result is None
    assert "aiod_utils[bioformats]" in str(record[0].message)


# ── Bio-Formats version pin and JVM configuration ───────────────────────────


def test_import_pins_bioformats_version(monkeypatch):
    monkeypatch.delenv("BIOFORMATS_VERSION", raising=False)
    importlib.reload(sys.modules["aiod_utils.io"])
    assert os.environ["BIOFORMATS_VERSION"] == BIOFORMATS_VERSION


def test_import_respects_existing_bioformats_version(monkeypatch):
    monkeypatch.setenv("BIOFORMATS_VERSION", "7.3.1")
    importlib.reload(sys.modules["aiod_utils.io"])
    assert os.environ["BIOFORMATS_VERSION"] == "7.3.1"


def test_configure_relocates_jgo_cache(fake_scyjava_config, monkeypatch, tmp_path):
    monkeypatch.setenv("JGO_CACHE_DIR", str(tmp_path))
    _configure_bioformats_jvm()
    assert fake_scyjava_config["cache_dir"] == str(tmp_path)


def test_configure_keeps_scyjava_default_cache_when_unset(
    fake_scyjava_config, monkeypatch
):
    monkeypatch.delenv("JGO_CACHE_DIR", raising=False)
    _configure_bioformats_jvm()
    assert "cache_dir" not in fake_scyjava_config


def test_configure_honours_java_fetch_override_without_vendor(
    fake_scyjava_config, monkeypatch
):
    # bffile ignores BFF_JAVA_FETCH unless a vendor/version is also set
    monkeypatch.delenv("BFF_JAVA_VENDOR", raising=False)
    monkeypatch.delenv("BFF_JAVA_VERSION", raising=False)
    monkeypatch.setenv("BFF_JAVA_FETCH", "never")
    _configure_bioformats_jvm()
    assert fake_scyjava_config["constraints"] == {"fetch": "never"}


@pytest.mark.skipif(
    not os.environ.get("AIOD_TEST_BIOFORMATS_FETCH"),
    reason="Downloads ~55MB of jars (and needs Java); set AIOD_TEST_BIOFORMATS_FETCH=1",
)
def test_prefetch_fills_relocated_caches_with_pinned_version(monkeypatch, tmp_path):
    pytest.importorskip("bioio_bioformats")
    jgo_cache, m2_repo = tmp_path / "jgo", tmp_path / "m2"
    monkeypatch.setenv("JGO_CACHE_DIR", str(jgo_cache))
    monkeypatch.setenv("M2_REPO", str(m2_repo))
    assert prefetch_bioformats() == os.environ["BIOFORMATS_VERSION"]
    assert any(jgo_cache.rglob("*.jar"))
    assert (m2_repo / "ome" / "formats-gpl" / os.environ["BIOFORMATS_VERSION"]).is_dir()
