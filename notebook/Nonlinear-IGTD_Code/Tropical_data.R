# R code: simulation function for the tropical distance-based model
simulate_tropical_igtd <- function(n         = 200,
                                   p         = 2500,
                                   S         = 5,
                                   Delta     = 2.2,
                                   sigma_seg = 0.05,
                                   sigma_n   = 0.01,
                                   center    = TRUE,
                                   seed      = NULL) {
  # n         : number of observations
  # p         : total number of features
  # S         : number of tropical segments among the first 100 features
  # Delta     : magnitude of class-1 shift for s_plus (+Delta) and s_minus (-Delta)
  # sigma_seg : within-segment noise sd for signal features
  # sigma_n   : noise sd for non-signal features
  # center    : whether to column-center the 100 signal features (recommended TRUE)
  # seed      : optional random seed
  
  if (!is.null(seed)) set.seed(seed)
  
  p_signal <- 100
  p_noise  <- p - p_signal
  if (S > p_signal) stop("S must be <= 100.")
  
  # ------------------------------------------------------
  # 1. Partition first 100 features into S random segments
  # ------------------------------------------------------
  signal_idx <- 1:p_signal
  shuffled   <- sample(signal_idx, p_signal, replace = FALSE)
  
  base_size <- floor(p_signal / S)
  seg_sizes <- rep(base_size, S)
  remainder <- p_signal - sum(seg_sizes)
  if (remainder > 0) {
    seg_sizes[1:remainder] <- seg_sizes[1:remainder] + 1
  }
  
  segments <- vector("list", S)
  start <- 1
  for (s in seq_len(S)) {
    end <- start + seg_sizes[s] - 1
    segments[[s]] <- shuffled[start:end]
    start <- end + 1
  }
  
  # ------------------------------------------------------
  # 2. Class labels
  # ------------------------------------------------------
  Y <- rbinom(n, size = 1, prob = 0.5)
  
  # ------------------------------------------------------
  # 3. Global s_plus and s_minus (fixed for whole dataset)
  # ------------------------------------------------------
  s_pm    <- sample(1:S, size = 2, replace = FALSE)
  s_plus  <- s_pm[1]
  s_minus <- s_pm[2]
  
  # ------------------------------------------------------
  # 4. Baseline segment values and class-dependent shifts
  # ------------------------------------------------------
  # baseline v_is ~ Uniform(-1,1)
  v_base <- matrix(runif(n * S, min = -1, max = 1), nrow = n, ncol = S)
  
  v_shifted <- v_base
  idx1 <- which(Y == 1)
  if (length(idx1) > 0) {
    v_shifted[idx1, s_plus]  <- v_shifted[idx1, s_plus]  + Delta
    v_shifted[idx1, s_minus] <- v_shifted[idx1, s_minus] - Delta
  }
  
  # ------------------------------------------------------
  # 5. Expand to 100 signal features + small within-segment noise
  # ------------------------------------------------------
  X_signal <- matrix(0, nrow = n, ncol = p_signal)
  for (s in seq_len(S)) {
    cols <- segments[[s]]
    X_signal[, cols] <- v_shifted[, s, drop = TRUE] +
      matrix(rnorm(n * length(cols), mean = 0, sd = sigma_seg),
             nrow = n, ncol = length(cols))
  }
  
  # Optional: column-wise centering (across samples)
  # This preserves tropical distance but weakens Euclidean/correlation cues.
  if (center) {
    X_signal <- scale(X_signal, center = TRUE, scale = FALSE)
  }
  
  # ------------------------------------------------------
  # 6. Noise features
  # ------------------------------------------------------
  if (p_noise > 0) {
    X_noise <- matrix(rnorm(n * p_noise, mean = 0, sd = sigma_n),
                      nrow = n, ncol = p_noise)
    X <- cbind(X_signal, X_noise)
  } else {
    X <- X_signal
  }
  
  colnames(X) <- paste0("X", seq_len(p))
  
  list(
    X        = X,          # predictors (n x p)
    Y        = Y,          # class labels (0/1)
    segments = segments,   # list of segment indices in 1:100
    s_plus   = s_plus,     # index of upward-shifted segment
    s_minus  = s_minus,    # index of downward-shifted segment
    params   = list(
      n         = n,
      p         = p,
      S         = S,
      Delta     = Delta,
      sigma_seg = sigma_seg,
      sigma_n   = sigma_n,
      center    = center
    )
  )
}
simG <- simulate_tropical_igtd(
  n         = 200,
  p         = 2500,
  S         = 5,
  Delta     = 1.6,   # moderate signal
  sigma_seg = 0.10,  # more within-segment noise (blurs Euclidean/correlation)
  sigma_n   = 0.03,  # stronger noise in non-signal features
  center    = TRUE,
  seed      = 123
)


