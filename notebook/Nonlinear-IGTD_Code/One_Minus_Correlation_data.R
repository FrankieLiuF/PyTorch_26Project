generate_rd_dominated_data <- function(n = 200, p = 2500, signal_dim = 100, rho = 0.95, signal_scale = 3) {
  # Load required library
  library(MASS)
  
  # 1. Construct Toeplitz correlation matrix for signal features
  toeplitz_cor <- function(p, rho) {
    mat <- matrix(0, nrow = p, ncol = p)
    for (i in 1:p) {
      for (j in 1:p) {
        mat[i, j] <- rho^abs(i - j)
      }
    }
    return(mat)
  }
  Sigma <- toeplitz_cor(signal_dim, rho)
  
  # 2. Generate multivariate normal signal features with high correlation
  X_signal <- mvrnorm(n = n, mu = rep(0, signal_dim), Sigma = Sigma)
  
  # 3. Construct structured beta vector (smooth increasing pattern)
  beta <- seq(1 / signal_dim, 1, length.out = signal_dim)
  
  # 4. Compute linear predictor and apply logistic function
  eta <- signal_scale * X_signal %*% beta
  probs <- 1 / (1 + exp(-eta))
  
  # 5. Simulate binary response
  Y <- rbinom(n, size = 1, prob = probs)
  
  # 6. Add Gaussian noise features to simulate high-dimensionality
  noise_dim <- p - signal_dim
  X_noise <- matrix(rnorm(n * noise_dim), nrow = n, ncol = noise_dim)
  
  # 7. Combine signal and noise features
  X <- cbind(X_signal, X_noise)
  
  return(list(X = X, Y = Y))
}

# Example usage:
set.seed(123)
sim_data <- generate_rd_dominated_data()
X <- sim_data$X
Y <- sim_data$Y
