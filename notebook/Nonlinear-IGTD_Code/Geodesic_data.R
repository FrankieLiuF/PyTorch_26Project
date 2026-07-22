set.seed(42)

# Parameters
n <- 200         # number of observations
p <- 2500        # total predictors
signal_dim <- 100
noise_dim <- p - signal_dim

# Step 1: Generate Swiss roll in 3D
swiss_roll_3d <- SwissRoll(N = n)

# Step 2: Project 3D Swiss roll to 100D using random linear projection
projection_matrix <- matrix(rnorm(3 * signal_dim), nrow = 3, ncol = signal_dim)
X_signal <- swiss_roll_3d %*% projection_matrix  # n x 100

# Step 3: Compute pairwise Euclidean distances
dist_matrix <- as.matrix(dist(X_signal))

# Step 4: Construct k-NN graph (e.g., k = 10)
k <- 10
adj_matrix <- matrix(0, n, n)
for (i in 1:n) {
  neighbors <- order(dist_matrix[i, ])[2:(k + 1)]  # Exclude self
  adj_matrix[i, neighbors] <- dist_matrix[i, neighbors]
}
# Symmetrize adjacency matrix
adj_matrix <- pmin(adj_matrix, t(adj_matrix))

# Step 5: Create graph and compute geodesic distances from a reference point
g <- graph_from_adjacency_matrix(adj_matrix, mode = "undirected", weighted = TRUE)
geo_dist <- shortest.paths(g, v = 1)  # Geodesic distance from point 1

# Step 6: Define nonlinear function f over geodesic distance
f <- function(d) sin(d) + 0.1 * d
f_values <- f(geo_dist)

# Step 7: Create binary response using threshold
tau <- median(f_values)
Y <- as.integer(f_values > tau)

# Step 8: Add white noise features
X_noise <- matrix(rnorm(n * noise_dim), nrow = n)
X <- cbind(X_signal, X_noise)

# Step 9: Output dataset
geodesic_dataset <- list(X = X, Y = Y)

# View structure
str(geodesic_dataset)