# Contributing to pytfce

Thank you for your interest in contributing to `pytfce`.

## Reporting issues

Please open a GitHub issue for:

- Bug reports (include a minimal reproducible example and your Python/OS version)
- Feature requests
- Questions about usage

## Development setup

```bash
git clone https://github.com/YOUR_USERNAME/pytfce.git
cd pytfce
pip install -e ".[all]"
pip install pytest
```

## Running tests

```bash
pytest tests/ -v
```

## Submitting changes

1. Fork the repository and create a feature branch from `main`.
2. Add tests for any new functionality.
3. Ensure all tests pass (`pytest tests/ -v`).
4. Open a pull request with a clear description of the change.

## Code style

- Follow PEP 8.
- Use NumPy-style docstrings for public functions.
- Keep dependencies minimal — core functionality should only require NumPy, SciPy, and `connected-components-3d`.

## Scope

`pytfce` focuses on probabilistic TFCE and related analytical inference methods for 3D volumetric neuroimaging data. Contributions that extend this scope (e.g., surface-based analysis, new inference frameworks) are welcome as proposals via GitHub issues before implementation.
