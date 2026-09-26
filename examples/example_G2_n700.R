# Reproducible G = 2, p = 3, n = 700 example matching the simulation design
# used in the manuscript. Run this script from the repository root.

source(file.path("R", "mpln_vem.R"))
compile_mpln_vem(file.path("src", "mpln_vem_core.cpp"))

simulate_manuscript_G2_n700 <- function(n = 700L, seed = 1L, time_range = c(0, 10)) {
  set.seed(seed)
  G <- 2L
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
      5.70,  0.25, -0.50,
      5.95,  0.65, -0.50,
      6.55, -0.30,  0.65
    ),
    c(
      5.75, -2.00,  1.60,
      5.90,  2.30,  1.55,
      6.30, -3.00,  1.80
    )
  )

  Omega_blocks <- list(
    list(
      matrix(c(0.80,  0.10,  0.10, 0.30), 2, 2, byrow = TRUE),
      matrix(c(0.65,  0.12,  0.12, 0.32), 2, 2, byrow = TRUE),
      matrix(c(0.75, -0.10, -0.10, 0.30), 2, 2, byrow = TRUE)
    ),
    list(
      matrix(c(0.70, -0.24, -0.24, 0.55), 2, 2, byrow = TRUE),
      matrix(c(0.85,  0.30,  0.30, 0.65), 2, 2, byrow = TRUE),
      matrix(c(0.68,  0.22,  0.22, 0.58), 2, 2, byrow = TRUE)
    )
  )
  Omega <- lapply(Omega_blocks, make_block_diag_omega)

  Sigma <- list(
    matrix(c(
       0.55,  0.18, -0.16,
       0.18,  0.50, -0.14,
      -0.16, -0.14,  0.48
    ), 3, 3, byrow = TRUE),
    matrix(c(
       0.58, -0.15,  0.16,
      -0.15,  0.55,  0.18,
       0.16,  0.18,  0.54
    ), 3, 3, byrow = TRUE)
  )

  # Class 2 is the reference class. Columns are intercept, c1, and c2.
  xi <- matrix(c(0.4, 1.0, -1.0), nrow = 1L)
  gating <- matrix(stats::rnorm(n * 2L), nrow = n, ncol = 2L,
                   dimnames = list(NULL, c("c1", "c2")))
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
      T_i <- sample(3:7, size = 1L)
      raw_time <- sort(stats::runif(
        T_i,
        min = time_range[1L],
        max = time_range[2L]
      ))
      
      if (T_i <= 1L || all(diff(raw_time) >= min_time_gap)) break
    }

    # The manuscript scales time to approximately [-1, 1]. The same scaled
    # time is used in both the fixed and random designs.
    time_i <- (raw_time - time_center) / time_scale
    x_i <- stats::rnorm(T_i, mean = 0.2 * time_i, sd = 0.1)
    fixed_i <- cbind("(Intercept)" = 1, time = time_i, x = x_i)
    Xbig_i <- expand_outcome_specific_design(fixed_i, p = p, T_i = T_i)
    M_i <- make_big_M(build_random_design(time_i, p))

    g <- true_class[i]
    b_i <- as.numeric(MASS::mvrnorm(1L, mu = rep(0, 2L * p),
                                    Sigma = Omega[[g]]))
    Y_i <- matrix(0, nrow = p, ncol = T_i)
    offset_i <- numeric(T_i)
    for (l in seq_len(T_i)) {
      rows <- ((l - 1L) * p + 1L):(l * p)
      epsilon_l <- as.numeric(MASS::mvrnorm(1L, mu = rep(0, p),
                                            Sigma = Sigma[[g]]))
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
      class = true_class
    )
  )
}

example <- simulate_manuscript_G2_n700(n = 700L, seed = 1L)

# Increase n_starts or max_iter for a more exhaustive analysis. The fit with
# the largest final ELBO is returned when n_starts > 1.
fit <- fit_mpln_vem(
  data = example$data,
  n_starts = 3L,
  start_seeds = 101:103,
  max_iter = 2000L,
  inner_cycles = 10L,
  tol = 1e-4,
  convergence = "absolute",
  init_method = "kmeans",
  verbose = TRUE
)

print(fit)
fit$parameter_count
fit$n_parameters          # 51 for this G = 2, p = 3 specification
fit$variational_BIC
fit$beta
fit$Omega
fit$Sigma
fit$gating_coefficients
head(fit$posterior_class_probabilities)
table(fitted = fit$class_assignment, truth = example$truth$class)
