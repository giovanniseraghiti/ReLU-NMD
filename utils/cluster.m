function S=cluster(picture)

% Generate a collection of points in different noisy clusters

% Cluster 1 (Central region)
mean_x1 = 0; mean_y1 = 0; mean_z1 = 0;
std_x1 = 3; std_y1 = 3; std_z1 =3;

% Cluster 2 (Upper-left region)
mean_x2 = -10; mean_y2 = 10; mean_z2 = 5;
std_x2 = 3; std_y2 = 3; std_z2 = 3;

% Cluster 3 (Bottom-right region)
mean_x3 = 10; mean_y3 = -10; mean_z3 = -10;
std_x3 = 3; std_y3 = 3; std_z3 = 3;

% Cluster 4 (Top-right region)
mean_x4 = 10; mean_y4 = 10; mean_z4 = 10;
std_x4 = 3; std_y4 = 3; std_z4 = 3;

% Cluster 5 (Bottom-left region)
mean_x5 = -10; mean_y5 = -10; mean_z5 = -10;
std_x5 = 3; std_y5 = 3; std_z5 = 3;

% Cluster 6 (Near the origin but with smaller spread)
mean_x6 = 5; mean_y6 = 5; mean_z6 = 5;
std_x6 = 3; std_y6 = 3; std_z6 = 3;

% Number of neurons to generate in each cluster
num_neurons1 = 30;
num_neurons2 = 30;
num_neurons3 = 30;
num_neurons4 = 40;
num_neurons5 = 30;
num_neurons6 = 40;

% Generate random neuron positions in each cluster using Gaussian distributions
x1 = mean_x1 + std_x1 * randn(num_neurons1, 1);
y1 = mean_y1 + std_y1 * randn(num_neurons1, 1);
z1 = mean_z1 + std_z1 * randn(num_neurons1, 1);

x2 = mean_x2 + std_x2 * randn(num_neurons2, 1);
y2 = mean_y2 + std_y2 * randn(num_neurons2, 1);
z2 = mean_z2 + std_z2 * randn(num_neurons2, 1);

x3 = mean_x3 + std_x3 * randn(num_neurons3, 1);
y3 = mean_y3 + std_y3 * randn(num_neurons3, 1);
z3 = mean_z3 + std_z3 * randn(num_neurons3, 1);

x4 = mean_x4 + std_x4 * randn(num_neurons4, 1);
y4 = mean_y4 + std_y4 * randn(num_neurons4, 1);
z4 = mean_z4 + std_z4 * randn(num_neurons4, 1);

x5 = mean_x5 + std_x5 * randn(num_neurons5, 1);
y5 = mean_y5 + std_y5 * randn(num_neurons5, 1);
z5 = mean_z5 + std_z5 * randn(num_neurons5, 1);

x6 = mean_x6 + std_x6 * randn(num_neurons6, 1);
y6 = mean_y6 + std_y6 * randn(num_neurons6, 1);
z6 = mean_z6 + std_z6 * randn(num_neurons6, 1);

% Combine all the neuron data from the different clusters
x = [x1; x2; x3; x4; x5; x6];
y = [y1; y2; y3; y4; y5; y6];
z = [z1; z2; z3; z4; z5; z6];
S=[x,y,z]';
if picture==1
    % Create a 3D scatter plot to visualize the neuron distribution
    figure;
    scatter3(x, y, z, 10, 'b','Marker','*','LineWidth',2); % 10 is the marker size, 'b' is the color (blue)
    grid on;
    axis equal;  % Keeps the aspect ratio equal for all axes
end

end