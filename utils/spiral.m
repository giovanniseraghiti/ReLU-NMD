function S=spiral(n,p,spir,picture)

%Creates a collection of points distributed on a spiral in a three-dimensional space

% Parameters for the spiral
numPoints=n;
turns = spir.turns; % Number of complete turns
radius_max = spir.radius_max; % Maximum radius of the spiral
height_max = spir.height_max; % Maximum height of the spiral

%perturbation
perturbation = p;

% Angle increment
theta = linspace(0, 2 * pi * turns, numPoints); % Angle parameter that increases with each point

% Generate spiral points in cylindrical coordinates
r = linspace(0, radius_max, numPoints); % Radius increases with theta
z = linspace(0, height_max, numPoints); % Height increases linearly along the spiral

% Convert to Cartesian coordinates
x = r .* cos(theta); % x = r * cos(theta)
y = r .* sin(theta); % y = r * sin(theta)

% Apply perturbation to each coordinate
x = x + perturbation * (rand(1, numPoints) - 0.5); % Random perturbation on x
y = y + perturbation * (rand(1, numPoints) - 0.5); % Random perturbation on y
z = z + perturbation * (rand(1, numPoints) - 0.5); % Random perturbation on z

% Combine coordinates into a matrix for plotting
points = [x', y', z'];
S=points';
if picture==1
    % Plot the perturbed spiral
    figure;
    plot3(x, y, z,'Marker','*','LineStyle','none');
    grid on;
    axis equal;
end
end