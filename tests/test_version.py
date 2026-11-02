"""Package version metadata and source-checkout fallback."""

import re
import runpy
from importlib import metadata

import pytfce


def test_version_is_public_and_looks_like_a_version():
    assert "__version__" in pytfce.__all__
    version = getattr(pytfce, "__version__", None)
    assert isinstance(version, str)
    assert re.fullmatch(r"\d+(?:\.\d+)*(?:[a-zA-Z0-9.+-]*)", version)


def test_version_comes_from_distribution_metadata(monkeypatch):
    def installed_version(name):
        assert name == "pytfce"
        return "1.2.3"

    monkeypatch.setattr(metadata, "version", installed_version)
    namespace = runpy.run_path(pytfce.__file__)
    assert namespace["__version__"] == "1.2.3"


def test_version_falls_back_without_distribution_metadata(monkeypatch):
    def missing_version(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", missing_version)
    namespace = runpy.run_path(pytfce.__file__)
    assert namespace["__version__"] == "0+unknown"
