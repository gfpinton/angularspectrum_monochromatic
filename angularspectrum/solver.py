import numpy as np
from scipy import fft


def angularspectrum_monochromatic_solver(initial_field, dx, dy, c0, f0, z_planes, max_angle_deg, alpha_dB_MHz_cm=0, boundary_factor=0.1, use_gpu=False):
    """
    Monochromatic angular spectrum wave propagation solver.
    
    Parameters:
    -----------
    initial_field : 2D complex array
        2D complex matrix [nx,ny] of initial pressure field
    dx, dy : float
        Grid spacing in x and y directions (m)
    c0 : float
        Speed of sound (m/s)
    f0 : float
        Frequency (Hz)
    z_planes : array
        Array of z positions where to calculate the field (m)
    max_angle_deg : float
        Maximum propagation angle (degrees)
    alpha_dB_MHz_cm : float, optional
        Attenuation coefficient in dB/(MHz*cm) (default = 0)
    boundary_factor : float, optional
        Thickness of absorbing boundary layer as fraction of domain (default = 0.1)
    use_gpu : bool, optional
        Flag to use GPU acceleration - ignored in this version (default = False)
        
    Returns:
    --------
    final_field : 2D complex array
        Final complex field at the last z plane
    amp_at_planes : 3D array
        Amplitude of field at each z plane [nx,ny,nz]
    """
    
    # Get dimensions
    nx, ny = initial_field.shape
    
    # Calculate wavenumber
    k0 = 2 * np.pi * f0 / c0
    
    # Prepare output arrays
    nz = len(z_planes)
    
    # Convert to single precision for better performance
    field = initial_field.astype(np.complex64)
    
    # Create spatial frequency grids (match MATLAB's fftshift behavior)
    fx = np.fft.fftshift(np.fft.fftfreq(nx, dx))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, dy))
    
    # Create 2D grid of wavenumbers
    KX, KY = np.meshgrid(2 * np.pi * fx, 2 * np.pi * fy, indexing='ij')
    
    # Calculate kz component using dispersion relation
    KZ_squared = k0**2 - KX**2 - KY**2
    
    # Apply angle constraint
    max_kxy = k0 * np.sin(np.deg2rad(max_angle_deg))
    KXY_mag = np.sqrt(KX**2 + KY**2)
    angle_filter = (KXY_mag <= max_kxy).astype(np.float32)
    
    # Create absorbing boundary layer to reduce reflections
    abl = create_boundary_layer(nx, ny, boundary_factor)
    
    # Allocate output arrays
    amp_at_planes = np.zeros((nx, ny, nz), dtype=np.float32)
    
    # Record initial field amplitude
    amp_at_planes[:, :, 0] = np.abs(field)
    
    # Calculate attenuation factors for each step
    attenuation_factors = np.ones(nz-1, dtype=np.float32)
    if alpha_dB_MHz_cm > 0:
        alpha_dB_m = alpha_dB_MHz_cm * (f0/1e6) * 100  # Convert to dB/m
        for i in range(1, nz):
            dz = z_planes[i] - z_planes[i-1]
            attenuation_factors[i-1] = 10**(-alpha_dB_m * dz / 20)  # Convert to amplitude ratio
    
    # Transform initial field to frequency domain
    field_freq = np.fft.fftshift(fft.fft2(field))
    
    # Loop through each z plane (starting from second, as first is initial)
    for i in range(1, nz):
        # Calculate propagation distance from previous plane
        dz = z_planes[i] - z_planes[i-1]
        
        # Create propagator for this step
        H = np.zeros((nx, ny), dtype=np.complex64)
        KZ = np.zeros((nx, ny), dtype=np.float32)
        
        # Propagating waves
        prop_waves = KZ_squared >= 0
        KZ[prop_waves] = np.sqrt(KZ_squared[prop_waves])
        H[prop_waves] = np.exp(1j * KZ[prop_waves] * dz)
        
        # Evanescent waves
        evan_waves = KZ_squared < 0
        KZ[evan_waves] = np.sqrt(-KZ_squared[evan_waves])
        H[evan_waves] = np.exp(-KZ[evan_waves] * dz)
        
        # Apply angle filter
        H = H * angle_filter
        
        # Apply propagator in frequency domain
        field_freq = field_freq * H
        
        # Transform back to spatial domain
        field = fft.ifft2(np.fft.ifftshift(field_freq))
        
        # Apply attenuation as a scalar factor for this step
        field = field * attenuation_factors[i-1]
        
        # Apply absorbing boundary layer to reduce reflections
        field = field * abl
        
        # Store amplitude at this plane
        amp_at_planes[:, :, i] = np.abs(field)
    
    # Final field is the field at the last z plane
    final_field = field
    
    return final_field, amp_at_planes


def create_boundary_layer(nx, ny, boundary_factor):
    """
    Create absorbing boundary layer to reduce reflections.
    
    Parameters:
    -----------
    nx, ny : int
        Dimensions of the field
    boundary_factor : float
        Thickness of boundary layer as a fraction of domain size
        
    Returns:
    --------
    abl : 2D array
        Absorbing boundary layer
    """
    # Create 1D boundary profiles
    x_boundary = create_boundary_vector(nx, round(nx * boundary_factor))
    y_boundary = create_boundary_vector(ny, round(ny * boundary_factor))
    
    # Create 2D boundary using outer product
    # Ensure correct dimensions [nx, ny]
    X = x_boundary[:, np.newaxis]  # Shape [nx, 1]
    Y = y_boundary[np.newaxis, :]  # Shape [1, ny]
    abl = X * Y  # Shape [nx, ny]
    
    return abl


def create_boundary_vector(n, thickness):
    """
    Create tapered boundary vector using a squared cosine window (Hann window).
    
    Parameters:
    -----------
    n : int
        Length of vector
    thickness : int
        Thickness of boundary in points
        
    Returns:
    --------
    vec : 1D array
        Boundary vector with tapered edges
    """
    vec = np.ones(n)
    
    # Apply taper at the edges using a squared cosine window (Hann window)
    for i in range(thickness):
        # Calculate normalized position from 0 to 1 (from edge to interior)
        x = (i + 0.5) / thickness
        
        # Squared cosine window value (0 at edge, 1 at interior)
        window_val = 0.5 * (1 - np.cos(np.pi * x))
        
        # Apply to both edges
        vec[i] = window_val
        vec[n-i-1] = window_val
    
    return vec 