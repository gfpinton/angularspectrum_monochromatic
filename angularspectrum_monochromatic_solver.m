function [final_field, amp_at_planes] = monochromatic_angular_spectrum_minimal(initial_field, dx, dy, c0, f0, z_planes, max_angle_deg, alpha_dB_MHz_cm, boundary_factor, use_gpu)
% MONOCHROMATIC_ANGULAR_SPECTRUM_MINIMAL - Minimal implementation of monochromatic wave propagation
%
% This stripped-down implementation focuses on the core propagation functionality
% with minimal extra features for easier debugging.
%
% INPUTS:
%   initial_field - 2D complex matrix [nx,ny] of initial pressure field
%   dx, dy - Grid spacing in x and y directions (m)
%   c0 - Speed of sound (m/s)
%   f0 - Frequency (Hz)
%   z_planes - Array of z positions where to calculate the field (m)
%   max_angle_deg - Maximum propagation angle (degrees)
%   alpha_dB_MHz_cm - Attenuation coefficient in dB/(MHz*cm) (optional, default = 0)
%   boundary_factor - Thickness of absorbing boundary layer as fraction of domain (optional, default = 0.1)
%   use_gpu - Flag to use GPU acceleration (optional, default = false)
%
% OUTPUTS:
%   final_field - Final complex field at the last z plane
%   amp_at_planes - Amplitude of field at each z plane [nx,ny,nz]

% Default attenuation to zero if not provided
if nargin < 8
    alpha_dB_MHz_cm = 0;
end

% Default boundary layer thickness to 10% of domain
if nargin < 9
    boundary_factor = 0.1;
end

% Default GPU usage to false
if nargin < 10
    use_gpu = false;
end

% Check if GPU is requested and available
if use_gpu
    try
        % Test if GPU is available by creating a small array
        test_gpu = gpuArray(single(1));
        clear test_gpu;
        
        % Check if arrays are large enough to benefit from GPU
        [nx, ny] = size(initial_field);
        total_elements = nx * ny;
        if total_elements < 1e6
            warning('Small arrays detected (%d elements). GPU acceleration may not be beneficial due to data transfer overhead.', total_elements);
            disp('Continuing with GPU acceleration as requested...');
        else
            disp('Using GPU acceleration for FFT operations');
        end
    catch
        warning('GPU requested but not available. Using CPU instead.');
        use_gpu = false;
    end
end

% Get dimensions
[nx, ny] = size(initial_field);

% Calculate wavenumber
k0 = 2*pi*f0/c0;

% Prepare output arrays
nz = length(z_planes);

% Convert to single precision for better GPU performance
field = single(initial_field);

% Create spatial frequency grids
fx = single((-nx/2:nx/2-1)/(nx*dx));  % Spatial frequencies in x (1/m)
fy = single((-ny/2:ny/2-1)/(ny*dy));  % Spatial frequencies in y (1/m)

% Create 2D grid of wavenumbers
% Note: For non-square domains, we need to ensure correct dimensionality
[KX, KY] = meshgrid(2*pi*fx, 2*pi*fy);  % Creates a [ny, nx] grid
KX = KX';  % Transpose to get [nx, ny] to match initial_field
KY = KY';  % Transpose to get [nx, ny] to match initial_field

% Calculate kz component using dispersion relation
KZ_squared = k0^2 - KX.^2 - KY.^2;

% Apply angle constraint
max_kxy = k0 * sin(deg2rad(max_angle_deg));
KXY_mag = sqrt(KX.^2 + KY.^2);
angle_filter = KXY_mag <= max_kxy;

% Create absorbing boundary layer to reduce reflections
% This tapers the field near the edges to avoid boundary reflections
abl = single(create_boundary_layer(nx, ny, boundary_factor));

% Move data to GPU if requested
if use_gpu
    KZ_squared = gpuArray(KZ_squared);
    angle_filter = gpuArray(angle_filter);
    abl = gpuArray(abl);
    field = gpuArray(field);
end

% Allocate output arrays - doing this after GPU decision to avoid unnecessary transfers
amp_at_planes = zeros(nx, ny, nz, 'single');

% Record initial field amplitude
amp_at_planes(:,:,1) = gather(abs(field));

% Transform initial field to frequency domain
% Use pagefun for large arrays on GPU
if use_gpu && (nx*ny > 1e6)
    field_freq = pagefun(@fft2, field);
else
    field_freq = fft2(field);
end

% Precompute propagators for all z steps if there are multiple planes
% This avoids redundant calculations in the loop
if nz > 2
    dz_steps = diff(z_planes);
    unique_dz_steps = unique(dz_steps);
    propagators = cell(length(unique_dz_steps), 1);
    
    for i = 1:length(unique_dz_steps)
        dz = unique_dz_steps(i);
        if use_gpu
            H = gpuArray.zeros(nx, ny, 'single');
            KZ = gpuArray.zeros(nx, ny, 'single');
        else
            H = zeros(nx, ny, 'single');
            KZ = zeros(nx, ny, 'single');
        end
        
        % Propagating waves
        prop_waves = KZ_squared >= 0;
        KZ(prop_waves) = sqrt(KZ_squared(prop_waves));
        H(prop_waves) = exp(1i * KZ(prop_waves) * dz);
        
        % Evanescent waves
        evan_waves = KZ_squared < 0;
        KZ(evan_waves) = sqrt(-KZ_squared(evan_waves));
        H(evan_waves) = exp(-KZ(evan_waves) * dz);
        
        % Apply angle filter
        H = H .* angle_filter;
        
        % Store precomputed propagator
        propagators{i} = fftshift(H);
    end
end

% Calculate attenuation factors for each step
attenuation_factors = ones(nz-1, 1, 'single');
if alpha_dB_MHz_cm > 0
    alpha_dB_m = alpha_dB_MHz_cm * (f0/1e6) * 100;  % Convert to dB/m
    for i = 2:nz
        dz = z_planes(i) - z_planes(i-1);
        attenuation_factors(i-1) = 10^(-alpha_dB_m * dz / 20);  % Convert to amplitude ratio
    end
end

% Loop through each z plane (starting from second, as first is initial)
for i = 2:nz
    % Calculate propagation distance from previous plane
    dz = z_planes(i) - z_planes(i-1);
    
    % Look up precomputed propagator if available
    if nz > 2
        [~, dz_idx] = min(abs(unique_dz_steps - dz));
        H_shifted = propagators{dz_idx};
    else
        % Create propagator for this step (with same dimensions as field)
        if use_gpu
            H = gpuArray.zeros(nx, ny, 'single');
            KZ = gpuArray.zeros(nx, ny, 'single');
        else
            H = zeros(nx, ny, 'single');
            KZ = zeros(nx, ny, 'single');
        end
        
        % Propagating waves
        prop_waves = KZ_squared >= 0;
        KZ(prop_waves) = sqrt(KZ_squared(prop_waves));
        H(prop_waves) = exp(1i * KZ(prop_waves) * dz);
        
        % Evanescent waves
        evan_waves = KZ_squared < 0;
        KZ(evan_waves) = sqrt(-KZ_squared(evan_waves));
        H(evan_waves) = exp(-KZ(evan_waves) * dz);
        
        % Apply angle filter
        H = H .* angle_filter;
        
        % Shift for FFT alignment
        H_shifted = fftshift(H);
    end
    
    % Apply propagator in frequency domain
    field_freq = field_freq .* H_shifted;
    
    % Transform back to spatial domain using pagefun for large arrays on GPU
    if use_gpu && (nx*ny > 1e6)
        field = pagefun(@ifft2, field_freq);
    else
        field = ifft2(field_freq);
    end
    
    % Apply attenuation as a scalar factor for this step
    field = field * attenuation_factors(i-1);
    
    % Apply absorbing boundary layer to reduce reflections
    field = field .* abl;
    
    % Store amplitude at this plane
    % Only gather from GPU when needed
    amp_at_planes(:,:,i) = gather(abs(field));
end

% Final field is the field at the last z plane
final_field = gather(field);
end

% Helper function to create absorbing boundary layer
function abl = create_boundary_layer(nx, ny, boundary_factor)
    % Create 1D boundary profiles
    x_boundary = create_boundary_vector(nx, round(nx * boundary_factor));
    y_boundary = create_boundary_vector(ny, round(ny * boundary_factor));
    
    % Create 2D boundary as outer product of 1D boundaries
    % Use meshgrid and transpose to ensure [nx, ny] dimensions
    [Y, X] = meshgrid(y_boundary, x_boundary);
    abl = X .* Y;  % Multiply together to get 2D tapering
end

% Create tapered boundary vector
function vec = create_boundary_vector(n, thickness)
    vec = ones(1, n);
    
    % Apply taper at the edges using a squared cosine window (Hann window)
    for i = 1:thickness
        % Calculate normalized position from 0 to 1 (from edge to interior)
        x = (i - 0.5) / thickness;
        
        % Squared cosine window value (0 at edge, 1 at interior)
        window_val = 0.5 * (1 - cos(pi * x));
        
        % Apply to both edges
        vec(i) = window_val;
        vec(n-i+1) = window_val;
    end
end