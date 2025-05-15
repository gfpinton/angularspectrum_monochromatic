# Angular Spectrum Monochromatic Wave Propagation

This repository contains both MATLAB and Python implementations of the Angular Spectrum method for monochromatic wave propagation. The code simulates how acoustic or electromagnetic waves propagate through homogeneous media.

## Features

- Monochromatic wave propagation using Angular Spectrum method
- GPU acceleration (MATLAB version and optional PyTorch in Python version)
- Absorbing boundary layers to reduce reflections
- Support for attenuation
- Maximum propagation angle constraint
- Example of focused beam simulation

## Repository Structure

- `matlab/` - Contains the original MATLAB implementation
- `angularspectrum/` - Python package implementation
- `test_minimal_solver_focused.py` - Python test script

## Python Implementation

### Requirements

- Python 3.6+
- NumPy
- SciPy
- Matplotlib
- PyTorch (optional, for GPU acceleration)

### Installation

```bash
pip install numpy scipy matplotlib
pip install torch  # Optional, for GPU acceleration

# Install from GitHub
pip install git+https://github.com/gfpinton/angularspectrum_monochromatic.git
```

### Usage

The main solver function is `angularspectrum_monochromatic_solver`, which takes an initial complex pressure field and propagates it through space:

```python
from angularspectrum import angularspectrum_monochromatic_solver

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

## MATLAB Implementation

The MATLAB implementation is in the `matlab/` directory and includes:

- `angularspectrum_monochromatic_solver.m` - The main solver function
- `test_minimal_solver_focused.m` - Test script for a focused beam

### Requirements

- MATLAB (tested with R2019b and later)
- Parallel Computing Toolbox (optional, for GPU acceleration)

### Usage

```matlab
% Example usage
[final_field, amp_at_planes] = angularspectrum_monochromatic_solver(initial_field, dx, dy, c0, f0, z_planes, max_angle_deg, alpha_dB_MHz_cm, boundary_factor, use_gpu);
```

## Example

Both implementations include an example script that demonstrates how to simulate a converging spherical wave and verify that it focuses at the expected focal distance.

To run the example in Python:

```bash
python test_minimal_solver_focused.py
```

This will generate a plot showing the initial field, the field at the focal plane, the maximum intensity versus propagation distance, and the beam profile at focus.

## License

[MIT License](LICENSE) 