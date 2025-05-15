# Angular Spectrum Monochromatic Wave Propagation

This repository contains a Python implementation of the Angular Spectrum method for monochromatic wave propagation. The code simulates how acoustic or electromagnetic waves propagate through homogeneous media.

## Features

- Monochromatic wave propagation using Angular Spectrum method
- GPU acceleration using PyTorch (optional)
- Absorbing boundary layers to reduce reflections
- Support for attenuation
- Maximum propagation angle constraint
- Example of focused beam simulation

## Requirements

- Python 3.6+
- NumPy
- SciPy
- Matplotlib
- PyTorch (optional, for GPU acceleration)

## Installation

```bash
pip install numpy scipy matplotlib
pip install torch  # Optional, for GPU acceleration
```

## Usage

The main solver function is `angularspectrum_monochromatic_solver`, which takes an initial complex pressure field and propagates it through space:

```python
from angularspectrum_monochromatic_solver import angularspectrum_monochromatic_solver

# Example usage
final_field, amp_at_planes = angularspectrum_monochromatic_solver(
    initial_field,  # 2D complex array [nx, ny]
    dx,             # Grid spacing in x (m)
    dy,             # Grid spacing in y (m)
    c0,             # Speed of sound (m/s)
    f0,             # Frequency (Hz)
    z_planes,       # Array of z positions to calculate field (m)
    max_angle_deg,  # Maximum propagation angle (degrees)
    alpha_dB_MHz_cm,# Attenuation coefficient (dB/MHz/cm)
    boundary_factor,# Thickness of absorbing boundary layer
    use_gpu         # Flag to use GPU acceleration
)
```

## Example

The repository includes an example script `test_minimal_solver_focused.py` that demonstrates how to simulate a converging spherical wave and verify that it focuses at the expected focal distance.

To run the example:

```bash
python test_minimal_solver_focused.py
```

This will generate a plot showing the initial field, the field at the focal plane, the maximum intensity versus propagation distance, and the beam profile at focus.

## Conversion from MATLAB

This code was converted from a MATLAB implementation. The main differences are:

1. Python uses 0-indexed arrays (vs. MATLAB's 1-indexed arrays)
2. NumPy/SciPy for numerical operations
3. PyTorch for GPU acceleration (vs. MATLAB's GPU arrays)
4. Matplotlib for plotting (vs. MATLAB's built-in plotting)

## License

[MIT License](LICENSE) 