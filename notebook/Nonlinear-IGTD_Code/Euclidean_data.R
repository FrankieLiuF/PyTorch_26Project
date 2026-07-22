generate_ed_data <- function(n = 200, p_signal = 100, p_noise = 2400, mu_beta = 2.5, sigma_x2 = 2) {
  set.seed(123)

  # Signal part
  Sigma_X <- diag(sigma_x2, p_signal)
  X_signal <- MASS::mvrnorm(n, mu = rep(0, p_signal), Sigma = Sigma_X)
  beta <- MASS::mvrnorm(1, mu = rep(mu_beta, p_signal), Sigma = diag(1, p_signal))
  eta <- X_signal %*% beta
  p <- 1 / (1 + exp(-eta))
  Y <- rbinom(n, 1, p)

  # Noise part
  X_noise <- matrix(rnorm(n * p_noise, mean = 0, sd = 1), nrow = n, ncol = p_noise)

  # Combine signal and noise
  X <- cbind(X_signal, X_noise)

  return(list(X = X, Y = Y))
}

# Example usage:
data <- generate_ed_data()
str(data)
