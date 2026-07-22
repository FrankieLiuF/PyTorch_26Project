generate_wasserstein_data <- function(n = 200, p = 2500, signal_dim = 100, seed = 123) {
  set.seed(seed)
  
  # Step 1: Define number of signal and noise features
  noise_dim <- p - signal_dim
  
  # Step 2: Simulate latent class labels (50% each class)
  Y <- rbinom(n, 1, 0.5)
  
  # Step 3: Generate signal features
  X_signal <- matrix(0, nrow = n, ncol = signal_dim)
  
  for (i in 1:n) {
    if (Y[i] == 0) {
      vec <- rnorm(signal_dim, mean = 0, sd = 1)
    } else {
      vec <- rnorm(signal_dim, mean = 1, sd = 1)
    }
    X_signal[i, ] <- sort(vec)  # Sorted to mimic quantile histograms
  }
  
  # Step 4: Create template distribution (mean of class 0)
  T_template <- colMeans(X_signal[Y == 0, , drop = FALSE])
  
  # Step 5: Define Wasserstein-1 distance (cumulative sum-based approximation)
  wasserstein_1d <- function(P, Q) {
    sum(abs(cumsum(P - Q)))
  }
  
  # Step 6: Compute Wasserstein distances for each observation
  W_dist <- apply(X_signal, 1, function(x) wasserstein_1d(x, T_template))
  
  # Step 7: Update labels based on distance from template
  threshold <- median(W_dist)
  Y_new <- as.integer(W_dist > threshold)
  
  # Step 8: Generate noise features
  X_noise <- matrix(rnorm(n * noise_dim), nrow = n, ncol = noise_dim)
  
  # Step 9: Combine signal and noise
  X <- cbind(X_signal, X_noise)
  
  # Step 10: Return as list
  return(list(X = X, Y = Y_new, Wasserstein = W_dist))
}


data <- generate_wasserstein_data()
dim(data$X)       # Should be 200 x 2500
table(data$Y)     # Balanced 0 and 1 classes based on W1 distance