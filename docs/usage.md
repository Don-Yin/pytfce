# Usage

## Installation

```bash
pip install pytfce
```

NumPy, SciPy and connected-components-3d install automatically. For a development install, see [CONTRIBUTING.md](../CONTRIBUTING.md).

## From a NIfTI file

Reading NIfTI needs NiBabel (`pip install nibabel`); the array API does not.

```python
import nibabel as nib
from pytfce import ptfce_baseline, ptfce_exact

img = nib.load("zstat1.nii.gz")
Z = img.get_fdata()
mask = Z != 0  # or load a proper brain mask

result = ptfce_baseline(Z, mask)  # fast, matches R pTFCE
result = ptfce_exact(Z, mask)     # hybrid eTFCE–GRF, exact cluster retrieval
```

To choose between them, see [Variants](../README.md#variants).

## Result

| Key | Contents |
|-----|----------|
| `p` | enhanced p-values (3-D array) |
| `logp` | −ln(p), natural log; divide by ln(10) for −log10(p) |
| `Z_enhanced` | enhanced Z-scores |
| `fwer_z_thresh` | GRF FWER Z-threshold for the enhanced map |
| `smoothness` | estimated smoothness dict |
| `diagnostics` | timings and run settings |
