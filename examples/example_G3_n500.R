# Reproducible G = 3, p = 2, n = 500 example matching the simulation design
# used in the manuscript. Run this script from the repository root.

source(file.path("R", "mpln_vem.R"))
compile_mpln_vem(file.path("src", "mpln_vem_core.cpp"))

simulate_manuscript_G3_n500 <- function(n = 500L, seed = 123L, time_range = c(0, 10)) {
  set.seed(seed)
  G <- 3L
  p <- 2L
  
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
      5.80,  1.10, -1.25,  0.65,
      6.45,  2.85, -1.20,  0.90
    ),
    c(
      5.90,  1.00, -1.15,  0.60,
      5.75, -1.85,  1.25, -0.90
    ),
    c(
      5.75,  1.05, -1.30,  0.70,
      6.15,  0.25,  1.85,  0.60
    )
  )

  Omega_blocks <- list(
    list(
      matrix(c(0.30, 0.05,
               0.05, 0.14), 2, 2, byrow = TRUE),
      matrix(c(0.34, 0.07,
               0.07, 0.18), 2, 2, byrow = TRUE)
    ),
    list(
      matrix(c(0.32,  0.04,
               0.04,  0.14), 2, 2, byrow = TRUE),
      matrix(c(0.34, -0.07,
              -0.07,  0.18), 2, 2, byrow = TRUE)
    ),
    list(
      matrix(c(0.30, 0.04,
               0.04, 0.13), 2, 2, byrow = TRUE),
      matrix(c(0.32, 0.06,
               0.06, 0.17), 2, 2, byrow = TRUE)
    )
  )
  Omega <- lapply(Omega_blocks, make_block_diag_omega)

  Sigma <- list(
    matrix(c(0.15, 0.040,
             0.04, 0.130), 2, 2, byrow = TRUE),
    matrix(c(0.15, 0.035,
             0.035, 0.13), 2, 2, byrow = TRUE),
    matrix(c(0.16, 0.04,
             0.04, 0.14), 2, 2, byrow = TRUE)
  )

  # Class 3 is the reference class. The columns are the intercept and z.
  xi <- matrix(
    c(
      -0.45,  1.25,
      -0.45, -1.25
    ),
    nrow = 2L,
    byrow = TRUE
  )
  z <- stats::rnorm(n)
  gating <- matrix(z, nrow = n, ncol = 1L,
                   dimnames = list(NULL, "z"))
  pi_true <- softmax_reference_G(gating, xi)
  true_class <- apply(pi_true, 1L, function(prob) {
    sample.int(G, size = 1L, prob = prob)
  })

  Y <- vector("list", n)
  random_time <- vector("list", n)
  X <- vector("list", n)
  offset <- vector("list", n)

  for (i in seq_len(n)) {
    repeat {
      T_i <- sample(2:5, size = 1L)
      raw_time <- sort(stats::runif(
        T_i,
        min = time_range[1L],
        max = time_range[2L]
      ))
      
      if (T_i <= 1L || all(diff(raw_time) >= min_time_gap)) break
    }

    # The time-invariant covariate z_i enters both the longitudinal model and
    # the gating network, as in the manuscript simulation.
    time_i <- (raw_time - time_center) / time_scale
    x_i <- stats::rnorm(T_i, mean = 0.2 * time_i, sd = 0.1)
    fixed_i <- cbind(
      "(Intercept)" = 1,
      time = time_i,
      x = x_i,
      z = rep(z[i], T_i)
    )
    Xbig_i <- expand_outcome_specific_design(fixed_i, p = p, T_i = T_i)
    M_i <- make_big_M(build_random_design(time_i, p))

    g <- true_class[i]
    b_i <- as.numeric(MASS::mvrnorm(
      1L,
      mu = rep(0, 2L * p),
      Sigma = Omega[[g]]
    ))
    Y_i <- matrix(0, nrow = p, ncol = T_i)
    offset_i <- numeric(T_i)

    for (l in seq_len(T_i)) {
      rows <- ((l - 1L) * p + 1L):(l * p)
      epsilon_l <- as.numeric(MASS::mvrnorm(
        1L,
        mu = rep(0, p),
        Sigma = Sigma[[g]]
      ))
      eta_l <- offset_i[l] +
        as.vector(Xbig_i[rows, , drop = FALSE] %*% beta[[g]]) +
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
    gating = gating,
    offset = offset,
    subject_id = paste0("subject", seq_len(n))
  )

  list(
    data = data,
    truth = list(
      beta = beta,
      Omega = Omega,
      Sigma = Sigma,
      gating_coefficients = xi,
      prior_class_probabilities = pi_true,
      class = true_class,
      z = z
    )
  )
}

example <- simulate_manuscript_G3_n500(n = 500L, seed = 17L)

# The G = 3 setting is more susceptible to local optima. As in the manuscript,
# retain the fit with the largest final ELBO among 20 independently seeded
# k-means-based initializations.
fit <- fit_mpln_vem(
  data = example$data,
  n_starts = 20L,
  start_seeds = 101:120,
  max_iter = 2000L,
  inner_cycles = 10L,
  tol = 1e-4,
  convergence = "absolute",
  init_method = "kmeans",
  verbose = TRUE
)

print(fit)
fit$start_summary
fit$parameter_count
fit$n_parameters          # 55 for this G = 3, p = 2 specification
fit$variational_BIC
fit$beta
fit$Omega
fit$Sigma
fit$gating_coefficients
head(fit$posterior_class_probabilities)
table(fitted = fit$class_assignment, truth = example$truth$class)
