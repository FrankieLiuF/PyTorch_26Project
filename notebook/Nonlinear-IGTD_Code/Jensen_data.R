# Load required package
if (!requireNamespace("MCMCpack", quietly = TRUE)) {
  install.packages("MCMCpack")
}
library(MCMCpack)

# Function to generate standard normal noise
generate_white_noise <- function(n, p) {
  matrix(rnorm(n * p), nrow = n, ncol = p)
}

# Function to generate Jensen-Shannon-based synthetic data (Option B)
generate_js_data_optionB <- function(n = 200, p = 2500) {
  # Number of signal features
  p_signal <- 100
  p_noise <- p - p_signal
  
  # Class labels (balanced binary)
  Y <- rbinom(n, 1, 0.5)
  
  # Dirichlet parameters
  alpha_0 <- rep(0.1, p_signal)   # Uniform-like for class 0
  alpha_1 <- rep(5, p_signal)   # More concentrated for class 1
  
  # Generate class-conditional signal features
  X_signal <- t(sapply(Y, function(y) {
    if (y == 0) {
      rdirichlet(1, alpha_0)
    } else {
      rdirichlet(1, alpha_1)
    }
  }))
  
  # Generate noise features
  X_noise <- generate_white_noise(n, p_noise)
  
  # Combine signal and noise
  X <- cbind(X_signal, X_noise)
  
  # Return as list
  return(list(X = X, Y = Y))
}

# Example usage
set.seed(123)
sim_data <- generate_js_data_optionB()
X <- sim_data$X
Y <- sim_data$Y

# Quick check
table(Y)
dim(X)
