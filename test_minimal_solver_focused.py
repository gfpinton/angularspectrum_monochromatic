import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import time
from angularspectrum import angularspectrum_monochromatic_solver

# Basic parameters
f0 = 8e6               # Frequency (Hz)
c0 = 1500              # Speed of sound (m/s)
lambda_val = c0/f0     # Wavelength (m)
k0 = 2*np.pi/lambda_val  # Wavenumber (rad/m)

# Domain parameters
wX = 25e-3             # Width in X direction (m)
wY = 14e-3             # Width in Y direction (m)
dX = lambda_val/5      # Grid spacing in X
dY = dX                # Grid spacing in Y
nX = round((wX+1.5e-2)/dX/2)*2+1  # Ensure odd for centerline
nY = round((wY+1.5e-2)/dY/2)*2+1

# Create spatial grid centered at origin
x = dX * np.arange(-(nX-1)/2, (nX-1)/2+1)
y = dY * np.arange(-(nY-1)/2, (nY-1)/2+1)

# Create aperture mask directly with the correct dimensions
aperture_mask = np.zeros((nX, nY))
for i in range(nX):
    for j in range(nY):
        if abs(x[i]) <= wX/2 and abs(y[j]) <= wY/2:
            aperture_mask[i, j] = 1

# Focal distance
focal_length = 3e-2    # Original focal length in meters
focal_wavelengths = focal_length/lambda_val  # Convert to wavelengths for reporting

# Create a converging spherical wave with the specified focal length
XX, YY = np.meshgrid(x, y, indexing='ij')
r = np.sqrt(XX**2 + YY**2)
path_diff = np.sqrt(r**2 + focal_length**2) - focal_length

# Phase for converging wave (negative sign because we want convergence)
phase = -k0 * path_diff

# Create the focused field (with aperture)
field = aperture_mask * np.exp(1j * phase)

# Normalize
field = field / np.max(np.abs(field))

# Define propagation distances (focused around the expected focal point)
z_distances = np.arange(0, focal_length*1.5, focal_length/50)

# Maximum propagation angle (degrees)
max_angle_deg = 85

# Run the minimal solver with attenuation and boundary layer
alpha_dB_MHz_cm = 0.5    # Moderate attenuation
boundary_factor = 0.25   # Moderate boundary layer

# Time the execution
start_time = time.time()
final_field, amp_at_planes = angularspectrum_monochromatic_solver(
    field, dX, dY, c0, f0, z_distances, max_angle_deg, alpha_dB_MHz_cm, boundary_factor
)
elapsed_time = time.time() - start_time
print(f"Execution time: {elapsed_time:.2f} seconds")

# Plot results
plt.figure(figsize=(12, 8))

# Plot the initial field amplitude
plt.subplot(2, 2, 1)
plt.imshow(np.abs(field).T, extent=[x[0]*1000, x[-1]*1000, y[0]*1000, y[-1]*1000])
plt.colorbar()
plt.title('Initial Field (Rectangular Aperture)')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.axis('equal')

# Find position of maximum intensity
max_intensity = np.zeros(len(z_distances))
for i in range(len(z_distances)):
    max_intensity[i] = np.max(amp_at_planes[:,:,i])

# Find the index with the maximum intensity (excluding the first plane)
if len(z_distances) > 1:
    max_idx = np.argmax(max_intensity[1:]) + 1
else:
    max_idx = 0

# Plot field amplitude at the focus
plt.subplot(2, 2, 2)
plt.imshow(amp_at_planes[:,:,max_idx].T, extent=[x[0]*1000, x[-1]*1000, y[0]*1000, y[-1]*1000])
plt.colorbar()
plt.title(f'Field at Focal Plane (z = {z_distances[max_idx]/lambda_val:.1f} λ)')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.axis('equal')

# Plot intensity vs. distance
plt.subplot(2, 2, 3)
plt.plot(z_distances/lambda_val, max_intensity, 'o-', linewidth=2)
plt.grid(True)
plt.title('Maximum Intensity vs. Propagation Distance')
plt.xlabel('Propagation Distance (wavelengths)')
plt.ylabel('Maximum Intensity')
plt.axvline(x=focal_wavelengths, color='r', linestyle='--', label='Expected Focus')
plt.legend()

# Plot field along centerline at focus
plt.subplot(2, 2, 4)
centerline_idx = (nY+1)//2 - 1  # Center row index (0-indexed in Python)
plt.plot(x*1000, amp_at_planes[:,centerline_idx,max_idx], linewidth=2)
plt.grid(True)
plt.title('Beam Profile at Focus')
plt.xlabel('x (mm)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.savefig('focused_beam_profile_python.png')
plt.show()

# Find the actual focal distance
max_idx = np.argmax(max_intensity)
actual_focal_distance = z_distances[max_idx]

print(f"Expected focal distance: {focal_wavelengths:.2f} wavelengths")
print(f"Actual focal distance: {actual_focal_distance/lambda_val:.2f} wavelengths")
print(f"Difference: {(actual_focal_distance - focal_length)/lambda_val:.2f} wavelengths")

# Calculate and display the relative error
rel_error = abs(actual_focal_distance - focal_length) / focal_length * 100
print(f"Relative error: {rel_error:.2f}%")
print(f"Attenuation: {alpha_dB_MHz_cm} dB/MHz/cm") 