# Reproducible G = 1, p = 3, n = 500 example matching the simulation design
# used in the manuscript. Run this script from the repository root.

source(file.path("R", "mpln_vem.R"))
compile_mpln_vem(file.path("src", "mpln_vem_core.cpp"))

simulate_manuscript_G1_n500 <- function(n = 500L, seed = 123L, time_range = c(0, 10)) {
  set.seed(seed)
  G <- 1L
  p <- 3L
  
  if (length(time_range) != 2L ||
      any(!is.finite(time_range)) ||
      time_range[1L] >= time_range[2L]) {
    stop(
      "time_range must contain two finite, increasing values.",
      call. = FALSE
    )
  }
  
  time_center <- mean(time_range)
  time_scale <- diff(time_range) / 2
  min_time_gap <- 0.05 * diff(time_range)

  beta <- list(
    c(
      6.00,  1.00, -1.50,
      5.80, -1.30,  1.20,
      7.00,  1.20,  0.80
    )
  )

  Omega_blocks <- list(
    matrix(c(1.00,  0.30,
             0.30,  1.00), 2, 2, byrow = TRUE),
    matrix(c(0.80,  0.20,
             0.20,  0.70), 2, 2, byrow = TRUE),
    matrix(c(0.60, -0.25,
            -0.25,  1.10), 2, 2, byrow = TRUE)
  )
  Omega <- list(make_block_diag_omega(Omega_blocks))

  Sigma <- list(
    matrix(c(
       0.50,  0.18, -0.16,
       0.18,  0.45, -0.14,
      -0.16, -0.14,  0.40
    ), 3, 3, byrow = TRUE)
  )

  Y <- vector("list", n)
  random_time <- vector("list", n)
  X <- vector("list", n)
  offset <- vector("list", n)

  for (i in seq_len(n)) {
    repeat {
      T_i <- sample(4:6, size = 1L)
      raw_time <- sort(stats::runif(
        T_i,
        min = time_range[1L],
        max = time_range[2L]
      ))
      
      if (T_i <= 1L || all(diff(raw_time) >= min_time_gap)) break
    }

    # The same centered and scaled time is used in the fixed- and
    # random-effect designs.
    time_i <- (raw_time - time_center) / time_scale
    x_i <- stats::rnorm(T_i, mean = 0.2 * time_i, sd = 0.1)
    fixed_i <- cbind("(Intercept)" = 1, time = time_i, x = x_i)
    Xbig_i <- expand_outcome_specific_design(fixed_i, p = p, T_i = T_i)
    M_i <- make_big_M(build_random_design(time_i, p))

    b_i <- as.numeric(MASS::mvrnorm(
      1L,
      mu = rep(0, 2L * p),
      Sigma = Omega[[1L]]
    ))
    Y_i <- matrix(0, nrow = p, ncol = T_i)
    offset_i <- numeric(T_i)

    for (l in seq_len(T_i)) {
      rows <- ((l - 1L) * p + 1L):(l * p)
      epsilon_l <- as.numeric(MASS::mvrnorm(
        1L,
        mu = rep(0, p),
        Sigma = Sigma[[1L]]
      ))
      eta_l <- offset_i[l] +
        as.vector(Xbig_i[rows, , drop = FALSE] %*% beta[[1L]]) +
        as.vector(M_i[rows, , drop = FALSE] %*% b_i) +
        epsilon_l
      Y_i[, l] <- stats::rpois(p, lambda = exp(eta_l))
    }

    Y[[i]] <- Y_i
    random_time[[i]] <- time_i
    X[[i]] <- fixed_i
    offset[[i]] <- offset_i
  }

  data <- make_mpln_data(
    Y = Y,
    time = random_time,
    X = X,
    G = G,
    gating = NULL,
    offset = offset,
    subject_id = paste0("subject", seq_len(n))
  )

  list(
    data = data,
    truth = list(
      beta = beta,
      Omega = Omega,
      Sigma = Sigma,
      prior_class_probabilities = matrix(1, nrow = n, ncol = 1L),
      class = rep(1L, n)
    )
  )
}

example <- simulate_manuscript_G1_n500(n = 500L, seed = 1L)

fit <- fit_mpln_vem(
  data = example$data,
  n_starts = 1L,
  start_seeds = 101L,
  max_iter = 2000L,
  inner_cycles = 10L,
  tol = 1e-4,
  convergence = "absolute",
  init_method = "kmeans",
  verbose = TRUE
)

print(fit)
fit$parameter_count
fit$n_parameters          # 24 for this G = 1, p = 3 specification
fit$variational_BIC
fit$beta
fit$Omega
fit$Sigma
head(fit$posterior_class_probabilities)

