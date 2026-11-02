# Contributing to pytfce

Thank you for your interest in contributing to `pytfce`.

## Reporting issues

Please open a GitHub issue for:

- Bug reports (include a minimal reproducible example and your Python/OS version)
- Feature requests
- Questions about usage

## Development setup

```bash
git clone https://github.com/Don-Yin/pytfce.git
cd pytfce
python -m pip install -e ".[all,test]"
python -m pip install ruff
```

## Running checks

Run Ruff from the repository root using the configuration in `pyproject.toml`:

```bash
python -m ruff check src/ tests/
```

Run the test suite against the editable source install:

```bash
python -m pytest tests/ -q -p no:cacheprovider
```

If the package is not installed in editable mode, prefix the test command with `PYTHONPATH=src`.

## Submitting changes

1. Fork the repository and create a feature branch from `main`.
2. Add tests for any new functionality.
3. Ensure the Ruff lint check and test suite above both pass.
4. Open a pull request with a clear description of the change.

## Code style

- Follow PEP 8.
- Use NumPy-style docstrings for public functions.
- Keep dependencies minimal — core functionality should only require NumPy, SciPy, and `connected-components-3d`.

## Scope

`pytfce` focuses on probabilistic TFCE and related analytical inference methods for 3D volumetric neuroimaging data. Contributions that extend this scope (e.g., surface-based analysis, new inference frameworks) are welcome as proposals via GitHub issues before implementation.
