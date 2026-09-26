# Lightweight integration checks for G = 1 and G = 2.
# Run from the repository root with: Rscript tests/smoke_test.R

source(file.path("R", "mpln_vem.R"))
compile_mpln_vem(file.path("src", "mpln_vem_core.cpp"))

simulate_small_mpln <- function(G, n = 18L, seed = 123L,
                                class_specific_design = FALSE) {
  set.seed(seed)
  p <- 2L
  gating <- cbind(c1 = stats::rnorm(n), c2 = stats::rnorm(n))
  xi <- if (G == 1L) matrix(numeric(0), 0L, 3L) else {
    matrix(stats::rnorm((G - 1L) * 3L, sd = 0.2), nrow = G - 1L)
  }
  pi <- if (G == 1L) matrix(1, n, 1L) else softmax_reference_G(gating, xi)
  z <- apply(pi, 1L, function(prob) sample.int(G, 1L, prob = prob))

  time <- lapply(seq_len(n), function(i) sort(stats::runif(3L, -1, 1)))
  X1 <- lapply(time, function(t) cbind("(Intercept)" = 1, time = t))
  if (class_specific_design && G > 1L) {
    X <- vector("list", G)
    X[[1L]] <- X1
    for (g in 2:G) {
      X[[g]] <- lapply(time, function(t) {
        cbind("(Intercept)" = 1, time = t, time2 = t^2)
      })
    }
  } else {
    X <- X1
  }

  q_beta <- if (class_specific_design && G > 1L) {
    c(2L * p, rep(3L * p, G - 1L))
  } else {
    rep(2L * p, G)
  }
  beta <- lapply(seq_len(G), function(g) {
    rep(c(1.0 + 0.1 * g, 0.15), length.out = q_beta[g])
  })
  Omega <- lapply(seq_len(G), function(g) {
    make_block_diag_omega(replicate(p, diag(c(0.08, 0.03)), simplify = FALSE))
  })
  Sigma <- lapply(seq_len(G), function(g) {
    matrix(c(0.10, 0.02, 0.02, 0.08), p, p)
  })
  offset <- lapply(seq_len(n), function(i) {
    log(stats::runif(length(time[[i]]), 0.8, 1.2))
  })

  Y <- vector("list", n)
  for (i in seq_len(n)) {
    g <- z[i]
    T_i <- length(time[[i]])
    X_i <- if (class_specific_design && G > 1L) X[[g]][[i]] else X[[i]]
    Xbig_i <- expand_outcome_specific_design(X_i, p, T_i)
    Mbig_i <- make_big_M(build_random_design(time[[i]], p))
    b_i <- as.numeric(MASS::mvrnorm(1L, rep(0, 2L * p), Omega[[g]]))
    eta <- numeric(p * T_i)
    for (l in seq_len(T_i)) {
      rows <- ((l - 1L) * p + 1L):(l * p)
      epsilon_l <- as.numeric(MASS::mvrnorm(1L, rep(0, p), Sigma[[g]]))
      eta[rows] <- rep(offset[[i]][l], p) +
        as.vector(Xbig_i[rows, , drop = FALSE] %*% beta[[g]]) +
        as.vector(Mbig_i[rows, , drop = FALSE] %*% b_i) +
        epsilon_l
    }
    Y[[i]] <- matrix(stats::rpois(p * T_i, exp(eta)), nrow = p)
  }

  make_mpln_data(
    Y = Y,
    time = time,
    X = X,
    G = G,
    gating = if (G == 1L) NULL else gating,
    offset = offset
  )
}

# G = 1: the gating network is bypassed, and the
# parameter count excludes all gating coefficients.
data_G1 <- simulate_small_mpln(G = 1L, seed = 101L)
stopifnot(count_mpln_parameters(data_G1)$total == 13L)
fit_G1 <- fit_mpln_vem(
  data_G1,
  n_starts = 1L,
  start_seeds = 1L,
  max_iter = 3L,
  inner_cycles = 2L,
  gating_maxit = 5L,
  verbose = FALSE
)
stopifnot(
  identical(dim(fit_G1$posterior_class_probabilities), c(data_G1$n, 1L)),
  all(fit_G1$posterior_class_probabilities == 1),
  nrow(fit_G1$gating_coefficients) == 0L,
  fit_G1$n_parameters == 13L,
  is.finite(fit_G1$variational_BIC)
)

# G = 2: exercise class-specific fixed designs with different dimensions,
# a two-covariate gating network, and nonzero known offsets.
data_G2 <- simulate_small_mpln(
  G = 2L,
  seed = 202L,
  class_specific_design = TRUE
)
stopifnot(identical(data_G2$q_beta, c(4L, 6L)))
stopifnot(count_mpln_parameters(data_G2)$total == 31L)
fit_G2 <- fit_mpln_vem(
  data_G2,
  n_starts = 1L,
  start_seeds = 2L,
  max_iter = 3L,
  inner_cycles = 2L,
  gating_maxit = 10L,
  verbose = FALSE
)
stopifnot(
  identical(
    unname(vapply(fit_G2$beta, length, integer(1))),
    c(4L, 6L)
  ),
  identical(dim(fit_G2$posterior_class_probabilities), c(data_G2$n, 2L)),
  max(abs(rowSums(fit_G2$posterior_class_probabilities) - 1)) < 1e-10,
  fit_G2$n_parameters == 31L,
  is.finite(fit_G2$variational_BIC)
)

message("All smoke tests passed.")
