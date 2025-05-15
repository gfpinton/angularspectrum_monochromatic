% Test script for the minimal monochromatic angular spectrum solver
% with a focused source
%
% This script creates a converging spherical wave and verifies
% that it focuses at the expected focal distance

clear;
close all;

% Basic parameters
f0 = 8e6;            % Frequency (Hz)
c0 = 1500;           % Speed of sound (m/s)
lambda = c0/f0;      % Wavelength (m)
k0 = 2*pi/lambda;    % Wavenumber (rad/m)

% Domain parameters
wX = 25e-3;          % Width in X direction (m)
wY = 14e-3;           % Width in Y direction (m)
dX = lambda/5;       % Grid spacing in X
dY = dX;             % Grid spacing in Y
nX = round((wX+1.5e-2)/dX/2)*2+1;  % Ensure odd for centerline
nY = round((wY+1.5e-2)/dY/2)*2+1;

% Create spatial grid centered at origin
x = dX * (-(nX-1)/2:(nX-1)/2);
y = dY * (-(nY-1)/2:(nY-1)/2);

% Create aperture mask directly with the correct dimensions
aperture_mask = zeros(nX, nY);
for i = 1:nX
    for j = 1:nY
        if abs(x(i)) <= wX/2 && abs(y(j)) <= wY/2
            aperture_mask(i,j) = 1;
        end
    end
end

% Focal distance
focal_length = 3e-2;  % Original focal length in meters
focal_wavelengths = focal_length/lambda;  % Convert to wavelengths for reporting

% Create a converging spherical wave with the specified focal length
[XX, YY] = meshgrid(x, y);
XX = XX'; % Transpose to get correct orientation [nX, nY]
YY = YY'; % Transpose to get correct orientation [nX, nY]
r = sqrt(XX.^2 + YY.^2);
path_diff = sqrt(r.^2 + focal_length^2) - focal_length;

% Phase for converging wave (negative sign because we want convergence)
phase = -k0 * path_diff;

% Create the focused field (with aperture)
field = aperture_mask .* exp(1i * phase);

% Normalize
field = field / max(abs(field(:)));

% Define propagation distances (focused around the expected focal point)
z_distances = [0:focal_length/50:focal_length*1.5];

% Maximum propagation angle (degrees)
max_angle_deg = 85;

% Run the minimal solver with attenuation and boundary layer
alpha_dB_MHz_cm = 0.5;   % Moderate attenuation
boundary_factor = 0.25;  % Moderate boundary layer

tic;
[final_field_gpu, amp_at_planes_gpu] = angularspectrum_monochromatic_solver(field, dX, dY, c0, f0, z_distances, max_angle_deg, alpha_dB_MHz_cm, boundary_factor, true);
toc

% Use GPU results if available, otherwise use CPU results
if exist('amp_at_planes_gpu', 'var')
    amp_at_planes = amp_at_planes_gpu;
    final_field = final_field_gpu;
end

% Plot results
figure('Position', [50, 50, 1200, 800]);

% Plot the initial field amplitude
subplot(2, 2, 1);
imagesc(x*1000, y*1000, abs(field)');
axis equal tight;
title('Initial Field (Rectangular Aperture)');
xlabel('x (mm)');
ylabel('y (mm)');
colorbar;

% Plot field amplitude at the focus
[~, max_idx] = max(squeeze(max(max(amp_at_planes, [], 1), [], 2)));
subplot(2, 2, 2);
imagesc(x*1000, y*1000, amp_at_planes(:,:,max_idx)');
axis equal tight;
title(['Field at Focal Plane (z = ' num2str(z_distances(max_idx)/lambda, '%.1f') ' λ)']);
xlabel('x (mm)');
ylabel('y (mm)');
colorbar;

% Find position of maximum intensity
max_intensity = zeros(length(z_distances), 1);
for i = 1:length(z_distances)
    max_intensity(i) = max(max(amp_at_planes(:,:,i)));
end

% Plot intensity vs. distance
subplot(2, 2, 3);
plot(z_distances/lambda, max_intensity, 'o-', 'LineWidth', 2);
grid on;
title('Maximum Intensity vs. Propagation Distance');
xlabel('Propagation Distance (wavelengths)');
ylabel('Maximum Intensity');
xline(focal_wavelengths, 'r--', 'Expected Focus');

% Plot field along centerline at focus
subplot(2, 2, 4);
centerline_idx = (nY+1)/2;  % Center row index
plot(x*1000, amp_at_planes(:,centerline_idx,max_idx), 'LineWidth', 2);
grid on;
title('Beam Profile at Focus');
xlabel('x (mm)');
ylabel('Amplitude');
saveas(gcf, 'focused_beam_profile.png');

% Find the actual focal distance
[~, max_idx] = max(max_intensity);
actual_focal_distance = z_distances(max_idx);

disp(['Expected focal distance: ' num2str(focal_wavelengths) ' wavelengths']);
disp(['Actual focal distance: ' num2str(actual_focal_distance/lambda) ' wavelengths']);
disp(['Difference: ' num2str((actual_focal_distance - focal_length)/lambda) ' wavelengths']);

% Calculate and display the relative error
rel_error = abs(actual_focal_distance - focal_length) / focal_length * 100;
disp(['Relative error: ' num2str(rel_error) '%']);
disp(['Attenuation: ' num2str(alpha_dB_MHz_cm) ' dB/MHz/cm']);
