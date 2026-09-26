# User-facing R interface for the latent-class Poisson-lognormal mixed model.
#
# The random-effects design is fixed to a random intercept and random time
# slope for each outcome. Fixed-effect designs may be common across classes or
# supplied separately for every class. The gating design may contain any set
# of subject-level covariates, with an intercept added internally.
#
# Main public functions
# ---------------------
# compile_mpln_vem(): compile and load src/mpln_vem_core.cpp.
# make_mpln_data(): validate user data and construct all internal designs.
# fit_mpln_vem(): fit one or more initializations and retain the largest ELBO.
#
# Data contract
# -------------
# Y[[i]] is a p-by-T_i count matrix and time[[i]] has length T_i. X may be
# either a list of n subject designs shared across classes or a list of G such
# lists. A T_i-by-q design is expanded to outcome-specific coefficients; users
# may instead supply a p*T_i-by-d matrix for a fully customized design. Rows of
# an expanded design and entries of a vector offset must be ordered by visit.
# offset[[i]] is a scalar or a length-T_i vector containing the known
# visit-level offsets o_il. The same offset is applied to all p outcomes at
# visit l. At a recorded visit all p outcomes must be observed; an entirely 
# unobserved visit should be omitted.

required_mpln_packages <- function() {
  c("Rcpp", "RcppArmadillo", "MASS", "nnet")
}

check_mpln_packages <- function(optional_mclust = FALSE) {
  packages <- required_mpln_packages()
  missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) {
    stop(
      "Install the required package(s) before fitting: ",
      paste(missing, collapse = ", "),
      call. = FALSE
    )
  }
  if (optional_mclust && !requireNamespace("mclust", quietly = TRUE)) {
    warning("Package 'mclust' is unavailable; initialization will use k-means.",
            call. = FALSE)
  }
  invisible(TRUE)
}

compile_mpln_vem <- function(cpp_file = file.path("src", "mpln_vem_core.cpp"),
                             rebuild = FALSE,
                             verbose = FALSE) {
  check_mpln_packages()
  if (!file.exists(cpp_file)) {
    stop("Cannot find the Rcpp source file: ", normalizePath(cpp_file, mustWork = FALSE),
         call. = FALSE)
  }
  Rcpp::sourceCpp(cpp_file, rebuild = rebuild, verbose = verbose)
  invisible(TRUE)
}

check_mpln_cpp_loaded <- function() {
  required <- c(
    "vstep_gmix_flexible_cpp",
    "mstep_gmix_flexible_cpp",
    "compute_B_gmix_flexible_cpp",
    "exact_elbo_gmix_flexible_cpp",
    "extract_variational_gmix_flexible_cpp"
  )
  missing <- required[!vapply(required, exists, logical(1), mode = "function",
                             inherits = TRUE)]
  if (length(missing)) {
    stop(
      "The Rcpp core is not loaded. Run compile_mpln_vem() before fit_mpln_vem().",
      call. = FALSE
    )
  }
  invisible(TRUE)
}

vectorise_by_visit <- function(A) as.vector(A)

build_random_design <- function(time_i, p) {
  time_i <- as.numeric(time_i)
  lapply(time_i, function(t) {
    out <- matrix(0, nrow = p, ncol = 2L * p)
    for (j in seq_len(p)) {
      cols <- (2L * j - 1L):(2L * j)
      out[j, cols] <- c(1, t)
    }
    out
  })
}

expand_outcome_specific_design <- function(X_i, p, T_i) {
  X_i <- as.matrix(X_i)
  storage.mode(X_i) <- "double"

  if (nrow(X_i) == p * T_i) {
    return(X_i)
  }
  if (nrow(X_i) != T_i) {
    stop(
      "Each fixed-effect design must have either T_i rows or p*T_i rows.",
      call. = FALSE
    )
  }

  q <- ncol(X_i)
  Xbig <- matrix(0, nrow = p * T_i, ncol = p * q)
  base_names <- colnames(X_i)
  if (is.null(base_names)) base_names <- paste0("x", seq_len(q))
  colnames(Xbig) <- unlist(lapply(seq_len(p), function(j) {
    paste0("outcome", j, ":", base_names)
  }), use.names = FALSE)

  for (l in seq_len(T_i)) {
    for (j in seq_len(p)) {
      row <- (l - 1L) * p + j
      cols <- ((j - 1L) * q + 1L):(j * q)
      Xbig[row, cols] <- X_i[l, ]
    }
  }
  Xbig
}

normalize_fixed_designs <- function(X, G, Y_list) {
  n <- length(Y_list)
  p <- nrow(Y_list[[1L]])

  is_subject_design_list <- function(z) {
    is.list(z) && length(z) == n &&
      all(vapply(z, function(x) is.matrix(x) || is.data.frame(x), logical(1)))
  }

  if (is_subject_design_list(X)) {
    X_by_class <- replicate(G, X, simplify = FALSE)
  } else if (is.list(X) && length(X) == G &&
             all(vapply(X, is_subject_design_list, logical(1)))) {
    X_by_class <- X
  } else {
    stop(
      "X must be either a list of n subject-level matrices or a list of G such lists.",
      call. = FALSE
    )
  }

  q_beta <- integer(G)
  coefficient_names <- vector("list", G)
  for (g in seq_len(G)) {
    X_by_class[[g]] <- lapply(seq_len(n), function(i) {
      expand_outcome_specific_design(X_by_class[[g]][[i]], p, ncol(Y_list[[i]]))
    })
    q_beta[g] <- ncol(X_by_class[[g]][[1L]])
    good <- vapply(X_by_class[[g]], ncol, integer(1)) == q_beta[g]
    if (!all(good)) {
      stop("All subjects must have the same coefficient dimension within class ", g, ".",
           call. = FALSE)
    }
    names_g <- colnames(X_by_class[[g]][[1L]])
    if (is.null(names_g)) names_g <- paste0("beta", seq_len(q_beta[g]))
    for (i in seq_len(n)) {
      current_names <- colnames(X_by_class[[g]][[i]])
      if (is.null(current_names)) {
        colnames(X_by_class[[g]][[i]]) <- names_g
      } else if (!identical(current_names, names_g)) {
        stop("Fixed-design column names or ordering differ across subjects in class ",
             g, ".", call. = FALSE)
      }
    }
    coefficient_names[[g]] <- names_g
  }
  list(
    X_by_class = X_by_class,
    q_beta = q_beta,
    coefficient_names = coefficient_names
  )
}

normalize_offsets <- function(offset, Y_list) {
  n <- length(Y_list)
  
  if (is.null(offset)) {
    return(lapply(Y_list, function(Y_i) {
      numeric(ncol(Y_i))
    }))
  }
  
  if (is.numeric(offset) && length(offset) == 1L) {
    if (!is.finite(offset)) {
      stop("The offset must be finite.", call. = FALSE)
    }
    return(lapply(Y_list, function(Y_i) {
      rep(as.numeric(offset), ncol(Y_i))
    }))
  }
  
  if (!is.list(offset) || length(offset) != n) {
    stop(
      "offset must be NULL, a scalar, or a list of length n.",
      call. = FALSE
    )
  }
  
  lapply(seq_len(n), function(i) {
    T_i <- ncol(Y_list[[i]])
    o_i <- as.numeric(offset[[i]])
    
    if (length(o_i) == 1L) {
      o_i <- rep(o_i, T_i)
    } else if (length(o_i) != T_i) {
      stop(
        "offset[[", i, "]] must be a scalar or a vector of length T_i = ",
        T_i, ".",
        call. = FALSE
      )
    }
    
    if (any(!is.finite(o_i))) {
      stop("All offsets must be finite.", call. = FALSE)
    }
    
    o_i
  })
}

expand_visit_offsets <- function(offset_list, p) {
  lapply(offset_list, function(o_i) {
    rep(o_i, each = p)
  })
}

make_mpln_data <- function(Y,
                           time,
                           X,
                           G,
                           gating = NULL,
                           offset = NULL,
                           subject_id = NULL) {
  if (length(G) != 1L || !is.finite(G) || G < 1L || G != as.integer(G)) {
    stop("G must be a positive integer.", call. = FALSE)
  }
  G <- as.integer(G)
  if (!is.list(Y) || !length(Y)) stop("Y must be a nonempty list.", call. = FALSE)
  n <- length(Y)
  if (!is.list(time) || length(time) != n) {
    stop("time must be a list of length n.", call. = FALSE)
  }

  Y_list <- lapply(seq_len(n), function(i) {
    Y_i <- as.matrix(Y[[i]])
    storage.mode(Y_i) <- "double"
    if (!nrow(Y_i) || !ncol(Y_i)) stop("Every subject must have at least one visit.", call. = FALSE)
    if (any(!is.finite(Y_i)) || any(Y_i < 0) || any(abs(Y_i - round(Y_i)) > 1e-8)) {
      stop("Y must contain finite, nonnegative integer counts with no NA values.", call. = FALSE)
    }
    Y_i
  })

  p <- nrow(Y_list[[1L]])
  if (!all(vapply(Y_list, nrow, integer(1)) == p)) {
    stop("All subjects must have the same number of outcomes p.", call. = FALSE)
  }
  outcome_names <- rownames(Y_list[[1L]])
  if (is.null(outcome_names)) outcome_names <- paste0("outcome", seq_len(p))
  for (i in seq_len(n)) {
    current_names <- rownames(Y_list[[i]])
    if (is.null(current_names)) {
      rownames(Y_list[[i]]) <- outcome_names
    } else if (!identical(current_names, outcome_names)) {
      stop("Outcome names or ordering differ across subjects.", call. = FALSE)
    }
  }

  time_list <- lapply(seq_len(n), function(i) {
    t_i <- as.numeric(time[[i]])
    if (length(t_i) != ncol(Y_list[[i]])) {
      stop("time[[", i, "]] must have one value per recorded visit.", call. = FALSE)
    }
    if (any(!is.finite(t_i))) stop("Visit times must be finite.", call. = FALSE)
    t_i
  })

  fixed <- normalize_fixed_designs(X, G, Y_list)
  offset_list <- normalize_offsets(offset, Y_list)
  offset_expanded_list <- expand_visit_offsets(offset_list, p)
  M_list <- lapply(time_list, build_random_design, p = p)

  if (is.null(gating)) {
    Cmat <- matrix(numeric(0), nrow = n, ncol = 0L)
  } else {
    Cmat <- as.matrix(gating)
    storage.mode(Cmat) <- "double"
    if (nrow(Cmat) != n) stop("gating must have n rows.", call. = FALSE)
    if (any(!is.finite(Cmat))) stop("gating must contain finite values.", call. = FALSE)
  }
  gating_names <- colnames(Cmat)
  
  if (is.null(gating_names)) {
    if (ncol(Cmat) == 0L) {
      gating_names <- character(0)
    } else {
      gating_names <- paste0("c", seq_len(ncol(Cmat)))
    }
  }
  
  colnames(Cmat) <- gating_names

  if (is.null(subject_id)) subject_id <- seq_len(n)
  if (length(subject_id) != n || anyDuplicated(subject_id)) {
    stop("subject_id must contain n unique values.", call. = FALSE)
  }

  out <- list(
    Y_list = Y_list,
    Xbig_by_class = fixed$X_by_class,
    M_list = M_list,
    offset_list = offset_list,
    offset_expanded_list = offset_expanded_list,
    Cmat = Cmat,
    time_list = time_list,
    subject_id = subject_id,
    n = n,
    p = p,
    r = 2L,
    G = G,
    qc = ncol(Cmat),
    q_beta = fixed$q_beta,
    coefficient_names = fixed$coefficient_names,
    outcome_names = outcome_names,
    gating_names = gating_names
  )
  class(out) <- "mpln_data"
  validate_mpln_data(out)
  out
}

validate_mpln_data <- function(data) {
  required <- c(
    "Y_list", "Xbig_by_class", "M_list",
    "offset_list", "offset_expanded_list",
    "Cmat", "time_list", "n", "p", "r", "G", "qc", "q_beta",
    "coefficient_names", "outcome_names", "gating_names"
  )
  missing <- setdiff(required, names(data))
  if (length(missing)) {
    stop("The data object is missing: ", paste(missing, collapse = ", "), call. = FALSE)
  }
  if (data$r != 2L) {
    stop("This implementation currently fixes the random design to intercept and slope.",
         call. = FALSE)
  }
  if (length(data$Y_list) != data$n ||
      length(data$M_list) != data$n ||
      length(data$offset_list) != data$n ||
      length(data$offset_expanded_list) != data$n ||
      nrow(data$Cmat) != data$n) {
    stop("Inconsistent subject counts in the data object.", call. = FALSE)
  }
  for (i in seq_len(data$n)) {
    T_i <- ncol(data$Y_list[[i]])
    
    if (length(data$offset_list[[i]]) != T_i) {
      stop(
        "offset_list[[", i, "]] must have length T_i.",
        call. = FALSE
      )
    }
    
    if (length(data$offset_expanded_list[[i]]) != data$p * T_i) {
      stop(
        "offset_expanded_list[[", i,
        "]] must have length p*T_i.",
        call. = FALSE
      )
    }
  }
  if (length(data$Xbig_by_class) != data$G) {
    stop("Xbig_by_class must have length G.", call. = FALSE)
  }
  invisible(TRUE)
}

normalize_rows <- function(M, eps = 1e-12) {
  M <- as.matrix(M)
  if (any(!is.finite(M)) || any(M < 0) || any(rowSums(M) <= 0)) {
    stop("Rows to be normalized must be finite, nonnegative, and have positive sums.",
         call. = FALSE)
  }
  M <- pmax(M, eps)
  M / rowSums(M)
}

softmax_rows <- function(logits) {
  logits <- as.matrix(logits)
  row_max <- apply(logits, 1L, max)
  ex <- exp(logits - row_max)
  ex / rowSums(ex)
}

softmax_reference_G <- function(Cmat, xi) {
  Cmat <- as.matrix(Cmat)
  xi <- as.matrix(xi)
  n <- nrow(Cmat)
  G <- nrow(xi) + 1L
  Xgate <- cbind(1, Cmat)
  if (ncol(xi) != ncol(Xgate)) {
    stop("xi has an incompatible number of columns.", call. = FALSE)
  }
  eta <- cbind(Xgate %*% t(xi), 0)
  pi <- softmax_rows(eta)
  colnames(pi) <- paste0("class", seq_len(G))
  pi
}

make_multinom_weights <- function(xi, qc) {
  xi <- as.matrix(xi)
  G <- nrow(xi) + 1L
  if (ncol(xi) != qc + 1L) {
    stop("xi has an incompatible number of columns.", call. = FALSE)
  }
  W <- matrix(0, nrow = G, ncol = qc + 2L)
  W[2:G, 2:(qc + 2L)] <- xi
  as.vector(t(W))
}

fit_gating_network <- function(Cmat,
                               tau,
                               maxit = 200L,
                               reltol = 1e-8,
                               decay = 1e-4,
                               trace = FALSE,
                               warm_start_xi = NULL) {
  Cmat <- as.matrix(Cmat)
  tau <- normalize_rows(tau)
  n <- nrow(tau)
  G <- ncol(tau)
  qc <- ncol(Cmat)

  if (G == 1L) {
    return(list(
      xi = matrix(0, nrow = 0L, ncol = qc + 1L),
      pi = matrix(1, nrow = n, ncol = 1L),
      fit = NULL,
      updated = TRUE
    ))
  }

  # Put the reference class first because nnet::multinom uses its first
  # response category as the baseline. The fitted probabilities and
  # coefficients are reordered back to classes 1,...,G below.
  Yresp <- tau[, c(G, seq_len(G - 1L)), drop = FALSE]
  colnames(Yresp) <- c(paste0("class", G), paste0("class", seq_len(G - 1L)))

  dat <- if (qc > 0L) as.data.frame(Cmat) else data.frame(.dummy = rep(1, n))
  if (qc > 0L) names(dat) <- paste0("c", seq_len(qc))
  form <- if (qc > 0L) {
    stats::as.formula(paste("Yresp ~", paste(names(dat), collapse = " + ")))
  } else {
    Yresp ~ 1
  }

  start_wts <- if (is.null(warm_start_xi)) NULL else {
    make_multinom_weights(warm_start_xi, qc)
  }
  max_wts <- max(1000L, (qc + 1L) * G * 10L,
                 if (is.null(start_wts)) 0L else length(start_wts))

  fit <- tryCatch(
    nnet::multinom(
      formula = form,
      data = dat,
      trace = trace,
      decay = decay,
      maxit = maxit,
      reltol = reltol,
      Hess = FALSE,
      MaxNWts = max_wts,
      Wts = start_wts,
      censored = FALSE,
      model = FALSE
    ),
    error = function(e) NULL
  )

  if (is.null(fit)) {
    xi <- if (is.null(warm_start_xi)) {
      matrix(0, nrow = G - 1L, ncol = qc + 1L)
    } else {
      as.matrix(warm_start_xi)
    }
    colnames(xi) <- c("(Intercept)", if (qc) paste0("c", seq_len(qc)))
    rownames(xi) <- paste0("class", seq_len(G - 1L))
    return(list(xi = xi, pi = softmax_reference_G(Cmat, xi),
                fit = NULL, updated = FALSE))
  }

  cf <- stats::coef(fit)
  if (is.null(dim(cf))) {
    cf <- matrix(cf, nrow = 1L, dimnames = list(NULL, names(cf)))
  } else {
    cf <- as.matrix(cf)
  }
  if (qc > 0L) {
    xi <- cf[, c("(Intercept)", names(dat)), drop = FALSE]
  } else {
    xi <- matrix(cf[, 1L], ncol = 1L,
                 dimnames = list(NULL, "(Intercept)"))
  }
  rownames(xi) <- paste0("class", seq_len(G - 1L))

  pi_raw <- as.matrix(stats::fitted(fit))
  if (ncol(pi_raw) != G) {
    stop("nnet::multinom returned an unexpected number of classes.", call. = FALSE)
  }
  pi <- matrix(0, nrow = n, ncol = G)
  pi[, G] <- pi_raw[, 1L]
  pi[, seq_len(G - 1L)] <- pi_raw[, 2:G, drop = FALSE]
  colnames(pi) <- paste0("class", seq_len(G))

  list(xi = xi, pi = pi, fit = fit, updated = TRUE)
}

make_block_diag_omega <- function(blocks) {
  p <- length(blocks)
  r <- nrow(blocks[[1L]])
  out <- matrix(0, p * r, p * r)
  for (j in seq_len(p)) {
    idx <- ((j - 1L) * r + 1L):(j * r)
    out[idx, idx] <- blocks[[j]]
  }
  out
}

spd_floor <- function(A, floor_value = 1e-8) {
  A <- 0.5 * (as.matrix(A) + t(as.matrix(A)))
  eg <- eigen(A, symmetric = TRUE)
  values <- pmax(eg$values, floor_value)
  out <- eg$vectors %*% diag(values, length(values)) %*% t(eg$vectors)
  0.5 * (out + t(out))
}

blockdiag_spd_floor <- function(A, p, r = 2L, floor_value = 1e-8) {
  out <- matrix(0, p * r, p * r)
  for (j in seq_len(p)) {
    idx <- ((j - 1L) * r + 1L):(j * r)
    out[idx, idx] <- spd_floor(A[idx, idx, drop = FALSE], floor_value)
  }
  0.5 * (out + t(out))
}

safe_solve_spd <- function(A, floor_value = 1e-8) {
  solve(spd_floor(A, floor_value))
}

safe_exp <- function(x, cap = 50) exp(pmax(pmin(x, cap), -cap))

make_big_M <- function(M_list_i) do.call(rbind, M_list_i)

build_prior_precision <- function(Mbig, Omega, Sigma, T_i) {
  db <- nrow(Omega)
  d_eta <- nrow(Mbig)
  Omega_inv <- safe_solve_spd(Omega)
  Sigma_inv <- safe_solve_spd(Sigma)
  Kinv <- kronecker(diag(T_i), Sigma_inv)
  Q <- matrix(0, db + d_eta, db + d_eta)
  Q[seq_len(db), seq_len(db)] <- Omega_inv + t(Mbig) %*% Kinv %*% Mbig
  eta_idx <- db + seq_len(d_eta)
  Q[seq_len(db), eta_idx] <- -t(Mbig) %*% Kinv
  Q[eta_idx, seq_len(db)] <- -Kinv %*% Mbig
  Q[eta_idx, eta_idx] <- Kinv
  0.5 * (Q + t(Q))
}

subject_feature_matrix <- function(data) {
  n <- data$n
  p <- data$p
  F <- matrix(0, nrow = n, ncol = 3L * p + data$qc)

  for (i in seq_len(n)) {
    Y_i <- data$Y_list[[i]]
    time_i <- data$time_list[[i]]
    offset_i <- matrix(data$offset_expanded_list[[i]], nrow = p)
    values <- numeric(0)

    for (j in seq_len(p)) {
      y <- log(Y_i[j, ] + 0.5) - offset_i[j, ]
      slope <- 0
      if (length(y) >= 2L && stats::sd(time_i) > 0) {
        slope <- unname(stats::coef(stats::lm(y ~ time_i))[2L])
        if (!is.finite(slope)) slope <- 0
      }
      y_sd <- if (length(y) >= 2L) stats::sd(y) else 0
      values <- c(values, mean(y), y_sd, slope)
    }
    if (data$qc > 0L) values <- c(values, data$Cmat[i, ])
    F[i, ] <- values
  }
  F[!is.finite(F)] <- 0
  F
}

repair_empty_clusters <- function(cluster, features, G) {
  cluster <- as.integer(cluster)
  counts <- tabulate(cluster, nbins = G)
  while (any(counts == 0L)) {
    empty <- which(counts == 0L)[1L]
    donor <- which.max(counts)
    donor_idx <- which(cluster == donor)
    if (length(donor_idx) <= 1L) {
      stop("Unable to repair an empty initial cluster.", call. = FALSE)
    }
    center <- colMeans(features[donor_idx, , drop = FALSE])
    distance <- rowSums((features[donor_idx, , drop = FALSE] -
                           matrix(center, nrow = length(donor_idx),
                                  ncol = ncol(features), byrow = TRUE))^2)
    moved <- donor_idx[which.max(distance)]
    cluster[moved] <- empty
    counts <- tabulate(cluster, nbins = G)
  }
  cluster
}

initial_clusters <- function(data,
                             method = c("kmeans", "mclust", "random"),
                             seed = NULL) {
  method <- match.arg(method)
  n <- data$n
  G <- data$G
  if (G == 1L) return(rep.int(1L, n))
  if (G > n) stop("G cannot exceed n.", call. = FALSE)
  if (!is.null(seed)) set.seed(seed)

  F <- subject_feature_matrix(data)
  F <- scale(F)
  F[!is.finite(F)] <- 0

  if (method == "random") {
    cluster <- sample(rep(seq_len(G), length.out = n))
  } else if (method == "mclust" && requireNamespace("mclust", quietly = TRUE)) {
    mc <- tryCatch(
      mclust::Mclust(F, G = G, verbose = FALSE),
      error = function(e) NULL
    )
    if (is.null(mc) || is.null(mc$classification)) {
      cluster <- stats::kmeans(F, centers = G, nstart = 50L,
                               iter.max = 100L)$cluster
    } else {
      cluster <- mc$classification
    }
  } else {
    if (method == "mclust") {
      warning("Package 'mclust' is unavailable; using k-means initialization.",
              call. = FALSE)
    }
    cluster <- stats::kmeans(F, centers = G, nstart = 50L,
                             iter.max = 100L)$cluster
  }
  repair_empty_clusters(cluster, F, G)
}

prepare_beta_initials <- function(beta_init, data) {
  if (is.null(beta_init)) return(NULL)
  G <- data$G

  if (is.list(beta_init) && length(beta_init) == G) {
    out <- lapply(beta_init, as.numeric)
  } else if (is.matrix(beta_init) && length(unique(data$q_beta)) == 1L &&
             all(dim(beta_init) == c(data$q_beta[1L], G))) {
    out <- lapply(seq_len(G), function(g) as.numeric(beta_init[, g]))
  } else {
    stop(
      "beta_init must be a list of length G, or a q_beta-by-G matrix when all classes use the same design dimension.",
      call. = FALSE
    )
  }

  for (g in seq_len(G)) {
    if (length(out[[g]]) != data$q_beta[g] || any(!is.finite(out[[g]]))) {
      stop("beta_init[[", g, "]] has the wrong length or non-finite values.",
           call. = FALSE)
    }
  }
  out
}

prepare_covariance_initials <- function(value, G, dimension, name) {
  if (is.null(value)) return(NULL)
  if (!is.list(value) || length(value) != G) {
    stop(name, " must be a list of length G.", call. = FALSE)
  }
  lapply(seq_len(G), function(g) {
    A <- as.matrix(value[[g]])
    if (!all(dim(A) == c(dimension, dimension)) || any(!is.finite(A))) {
      stop(name, "[[", g, "]] has incompatible dimensions or non-finite values.",
           call. = FALSE)
    }
    A
  })
}

fit_initial_beta <- function(data, subjects, g, ridge = 1e-4) {
  qg <- data$q_beta[g]
  lhs <- matrix(0, qg, qg)
  rhs <- numeric(qg)

  for (i in subjects) {
    X_i <- data$Xbig_by_class[[g]][[i]]
    response <- log(vectorise_by_visit(data$Y_list[[i]]) + 0.5) -
      data$offset_expanded_list[[i]]
    lhs <- lhs + crossprod(X_i)
    rhs <- rhs + as.numeric(crossprod(X_i, response))
  }
  tryCatch(
    as.numeric(solve(lhs + ridge * diag(qg), rhs)),
    error = function(e) rep(0, qg)
  )
}

estimate_initial_covariances <- function(data, clusters, beta_list,
                                         ridge = 1e-4) {
  G <- data$G
  p <- data$p
  r <- data$r
  db <- p * r
  Omega <- vector("list", G)
  Sigma <- vector("list", G)

  for (g in seq_len(G)) {
    members <- which(clusters == g)
    if (!length(members)) members <- seq_len(data$n)
    Bhat <- matrix(NA_real_, nrow = length(members), ncol = db)
    visit_residuals <- matrix(numeric(0), nrow = 0L, ncol = p)

    for (a in seq_along(members)) {
      i <- members[a]
      X_i <- data$Xbig_by_class[[g]][[i]]
      M_i <- make_big_M(data$M_list[[i]])
      response <- log(vectorise_by_visit(data$Y_list[[i]]) + 0.5) -
        data$offset_expanded_list[[i]]
      residual <- response - as.vector(X_i %*% beta_list[[g]])
      b_i <- tryCatch(
        as.numeric(solve(crossprod(M_i) + ridge * diag(db),
                         crossprod(M_i, residual))),
        error = function(e) rep(0, db)
      )
      Bhat[a, ] <- b_i
      visit_residuals <- rbind(
        visit_residuals,
        matrix(residual - as.vector(M_i %*% b_i), ncol = p, byrow = TRUE)
      )
    }

    Omega_g <- matrix(0, db, db)
    for (j in seq_len(p)) {
      idx <- ((j - 1L) * r + 1L):(j * r)
      block <- if (nrow(Bhat) >= 2L) {
        stats::cov(Bhat[, idx, drop = FALSE])
      } else {
        diag(c(0.20, 0.05))
      }
      if (!is.matrix(block)) block <- matrix(block, r, r)
      Omega_g[idx, idx] <- spd_floor(block + 0.05 * diag(r), 1e-6)
    }
    Omega[[g]] <- blockdiag_spd_floor(Omega_g, p, r, 1e-6)

    Sigma[[g]] <- if (nrow(visit_residuals) >= max(2L, p + 1L)) {
      S <- stats::cov(visit_residuals)
      if (!is.matrix(S)) S <- matrix(S, p, p)
      spd_floor(S + 0.02 * diag(p), 1e-6)
    } else {
      diag(0.10, p)
    }
  }
  list(Omega = Omega, Sigma = Sigma)
}

initialize_population <- function(data,
                                  beta_init = NULL,
                                  Omega_init = NULL,
                                  Sigma_init = NULL,
                                  tau_init = NULL,
                                  init_method = c("kmeans", "mclust", "random"),
                                  init_seed = NULL,
                                  ridge = 1e-4) {
  init_method <- match.arg(init_method)
  G <- data$G
  p <- data$p
  r <- data$r

  beta_user <- prepare_beta_initials(beta_init, data)
  Omega_user <- prepare_covariance_initials(Omega_init, G, p * r, "Omega_init")
  Sigma_user <- prepare_covariance_initials(Sigma_init, G, p, "Sigma_init")

  if (!is.null(tau_init)) {
    tau_init <- as.matrix(tau_init)
    if (!all(dim(tau_init) == c(data$n, G))) {
      stop("tau_init must have dimension n by G.", call. = FALSE)
    }
    clusters <- max.col(normalize_rows(tau_init), ties.method = "first")
  } else {
    clusters <- initial_clusters(data, init_method, init_seed)
  }

  if (is.null(beta_user)) {
    beta_list <- lapply(seq_len(G), function(g) {
      members <- which(clusters == g)
      if (length(members) < 2L) members <- seq_len(data$n)
      beta_g <- fit_initial_beta(data, members, g, ridge)
      if (!is.null(init_seed)) set.seed(init_seed + 100L + g)
      beta_g + stats::rnorm(length(beta_g), sd = if (G == 1L) 0 else 0.02)
    })
  } else {
    beta_list <- beta_user
  }

  estimated <- estimate_initial_covariances(data, clusters, beta_list, ridge)
  Omega <- if (is.null(Omega_user)) estimated$Omega else {
    lapply(Omega_user, blockdiag_spd_floor, p = p, r = r, floor_value = 1e-8)
  }
  Sigma <- if (is.null(Sigma_user)) estimated$Sigma else {
    lapply(Sigma_user, spd_floor, floor_value = 1e-8)
  }

  list(beta = beta_list, Omega = Omega, Sigma = Sigma, clusters = clusters)
}

initialize_variational <- function(data, beta, Omega, Sigma, exp_cap = 50) {
  G <- data$G
  n <- data$n
  p <- data$p
  r <- data$r
  db <- p * r
  m_joint <- vector("list", G)
  C_joint <- vector("list", G)

  for (g in seq_len(G)) {
    m_g <- vector("list", n)
    C_g <- vector("list", n)
    for (i in seq_len(n)) {
      Y_i <- data$Y_list[[i]]
      T_i <- ncol(Y_i)
      d_eta <- p * T_i
      M_i <- make_big_M(data$M_list[[i]])
      x_i <- data$offset_expanded_list[[i]] +
        as.vector(data$Xbig_by_class[[g]][[i]] %*% beta[[g]])
      Q <- build_prior_precision(M_i, Omega[[g]], Sigma[[g]], T_i)
      m0 <- c(rep(0, db), x_i)
      eta_idx <- db + seq_len(d_eta)
      precision <- Q
      precision[eta_idx, eta_idx] <- precision[eta_idx, eta_idx] +
        diag(safe_exp(x_i, exp_cap), d_eta)
      m_g[[i]] <- m0
      C_g[[i]] <- safe_solve_spd(precision)
    }
    m_joint[[g]] <- m_g
    C_joint[[g]] <- C_g
  }
  list(m_joint = m_joint, C_joint = C_joint)
}

count_mpln_parameters <- function(data) {
  G <- data$G
  p <- data$p
  r <- data$r
  fixed <- sum(data$q_beta)
  random_covariance <- G * p * r * (r + 1L) / 2L
  visit_covariance <- G * p * (p + 1L) / 2L
  gating <- if (G == 1L) 0L else (G - 1L) * (data$qc + 1L)

  breakdown <- c(
    fixed_effects = fixed,
    random_effect_covariances = random_covariance,
    visit_level_covariances = visit_covariance,
    gating_network = gating
  )
  list(total = as.integer(sum(breakdown)), breakdown = breakdown)
}

elbo_change <- function(current, previous, type = c("absolute", "relative")) {
  type <- match.arg(type)
  if (!is.finite(previous)) return(Inf)
  change <- abs(current - previous)
  if (type == "relative") change <- change / max(1, abs(previous))
  change
}

fit_mpln_vem_once <- function(data,
                              beta_init = NULL,
                              Omega_init = NULL,
                              Sigma_init = NULL,
                              xi_init = NULL,
                              tau_init = NULL,
                              max_iter = 500L,
                              inner_cycles = 10L,
                              max_newton = 1L,
                              max_backtrack = 10L,
                              max_fixed = 1L,
                              tol = 1e-4,
                              convergence = c("absolute", "relative"),
                              tol_inner = 1e-4,
                              diag_perturb = 0.25,
                              rcond_tol = 0,
                              exp_cap = 50,
                              gating_maxit = 200L,
                              gating_reltol = 1e-8,
                              gating_decay = 1e-4,
                              weight_floor = 1e-8,
                              init_method = c("kmeans", "mclust", "random"),
                              init_seed = NULL,
                              verbose = TRUE) {
  check_mpln_packages(optional_mclust = identical(init_method[1L], "mclust"))
  check_mpln_cpp_loaded()
  validate_mpln_data(data)
  convergence <- match.arg(convergence)
  init_method <- match.arg(init_method)

  integer_controls <- c(
    max_iter = max_iter,
    inner_cycles = inner_cycles,
    max_newton = max_newton,
    max_backtrack = max_backtrack,
    max_fixed = max_fixed,
    gating_maxit = gating_maxit
  )
  if (any(!is.finite(integer_controls)) || any(integer_controls < 1) ||
      any(integer_controls != as.integer(integer_controls))) {
    stop("Iteration controls must be positive integers.", call. = FALSE)
  }
  if (!is.finite(tol) || tol <= 0 || !is.finite(tol_inner) || tol_inner <= 0) {
    stop("tol and tol_inner must be positive finite values.", call. = FALSE)
  }
  if (!is.finite(weight_floor) || weight_floor <= 0 || weight_floor >= 1) {
    stop("weight_floor must lie strictly between 0 and 1.", call. = FALSE)
  }

  G <- data$G
  n <- data$n
  p <- data$p
  r <- data$r
  qc <- data$qc

  population <- initialize_population(
    data = data,
    beta_init = beta_init,
    Omega_init = Omega_init,
    Sigma_init = Sigma_init,
    tau_init = tau_init,
    init_method = init_method,
    init_seed = init_seed
  )
  beta <- population$beta
  Omega <- population$Omega
  Sigma <- population$Sigma

  if (!is.null(tau_init)) {
    tau <- as.matrix(tau_init)
    if (!all(dim(tau) == c(n, G))) {
      stop("tau_init must have dimension n by G.", call. = FALSE)
    }
    tau <- normalize_rows(tau)
  } else {
    tau <- matrix(1 / G, nrow = n, ncol = G)
  }

  if (G == 1L) {
    xi <- matrix(0, nrow = 0L, ncol = qc + 1L)
    pi <- matrix(1, nrow = n, ncol = 1L)
    gating_fit <- NULL
  } else if (is.null(xi_init)) {
    tau_gate <- if (!is.null(tau_init)) {
      tau
    } else {
      out <- matrix(0.02 / G, nrow = n, ncol = G)
      out[cbind(seq_len(n), population$clusters)] <-
        out[cbind(seq_len(n), population$clusters)] + 0.98
      normalize_rows(out)
    }
    gate <- fit_gating_network(
      data$Cmat, tau_gate,
      maxit = gating_maxit,
      reltol = gating_reltol,
      decay = gating_decay
    )
    xi <- gate$xi
    pi <- gate$pi
    gating_fit <- gate$fit
  } else {
    xi <- as.matrix(xi_init)
    if (!all(dim(xi) == c(G - 1L, qc + 1L))) {
      stop("xi_init must have dimension (G-1) by (q_c+1).", call. = FALSE)
    }
    pi <- softmax_reference_G(data$Cmat, xi)
    gating_fit <- NULL
  }

  variational <- initialize_variational(data, beta, Omega, Sigma, exp_cap)
  elbo_trace <- rep(NA_real_, max_iter)
  previous_elbo <- -Inf
  converged <- FALSE
  B <- matrix(NA_real_, nrow = n, ncol = G)
  completed_iterations <- 0L

  gate_iterations <- function(iteration) {
    if (iteration <= 10L) return(min(25L, gating_maxit))
    if (iteration <= 30L) return(min(75L, gating_maxit))
    gating_maxit
  }

  timing <- system.time({
    for (iteration in seq_len(max_iter)) {
      vstep <- vstep_gmix_flexible_cpp(
        Y_list = data$Y_list,
        Xbig_by_class = data$Xbig_by_class,
        M_list = data$M_list,
        offset_list = data$offset_expanded_list,
        beta_list = beta,
        Omega_list = Omega,
        Sigma_list = Sigma,
        m_joint_list = variational$m_joint,
        C_joint_list = variational$C_joint,
        p = p,
        r = r,
        inner_cycles = inner_cycles,
        max_newton = max_newton,
        max_backtrack = max_backtrack,
        max_fixed = max_fixed,
        tol_inner = tol_inner,
        rcond_tol = rcond_tol,
        diag_perturb = diag_perturb,
        exp_cap = exp_cap
      )
      variational$m_joint <- vstep$m_joint_list
      variational$C_joint <- vstep$C_joint_list

      tau <- softmax_rows(log(pmax(pi, 1e-300)) + as.matrix(vstep$B))

      mstep <- mstep_gmix_flexible_cpp(
        Xbig_by_class = data$Xbig_by_class,
        M_list = data$M_list,
        offset_list = data$offset_expanded_list,
        tau = tau,
        m_joint_list = variational$m_joint,
        C_joint_list = variational$C_joint,
        beta_list_current = beta,
        Omega_list_current = Omega,
        Sigma_list_current = Sigma,
        p = p,
        r = r,
        weight_floor = weight_floor
      )
      beta <- mstep$beta_list
      Omega <- mstep$Omega_list
      Sigma <- mstep$Sigma_list

      if (G > 1L) {
        gate <- fit_gating_network(
          data$Cmat, tau,
          maxit = gate_iterations(iteration),
          reltol = gating_reltol,
          decay = gating_decay,
          warm_start_xi = xi
        )
        xi <- gate$xi
        pi <- gate$pi
        gating_fit <- gate$fit
      }

      B <- compute_B_gmix_flexible_cpp(
        Y_list = data$Y_list,
        Xbig_by_class = data$Xbig_by_class,
        M_list = data$M_list,
        offset_list = data$offset_expanded_list,
        beta_list = beta,
        Omega_list = Omega,
        Sigma_list = Sigma,
        m_joint_list = variational$m_joint,
        C_joint_list = variational$C_joint,
        p = p,
        r = r,
        exp_cap = exp_cap
      )
      current_elbo <- exact_elbo_gmix_flexible_cpp(tau = tau, pi = pi, B = B)
      elbo_trace[iteration] <- current_elbo
      completed_iterations <- iteration

      change <- elbo_change(current_elbo, previous_elbo, convergence)
      if (verbose) {
        cat(sprintf(
          "Iteration %d: ELBO = %.8f; %s change = %.3e\n",
          iteration, current_elbo, convergence, change
        ))
      }
      if (iteration > 1L && is.finite(change) && change < tol) {
        converged <- TRUE
        break
      }
      previous_elbo <- current_elbo
    }
  })

  elbo_trace <- elbo_trace[seq_len(completed_iterations)]
  variational_conditional <- extract_variational_gmix_flexible_cpp(
    Y_list = data$Y_list,
    m_joint_list = variational$m_joint,
    C_joint_list = variational$C_joint,
    p = p,
    r = r
  )

  parameter_count <- count_mpln_parameters(data)
  final_elbo <- tail(elbo_trace, 1L)
  variational_BIC <- -2 * final_elbo + parameter_count$total * log(n)
  class_names <- paste0("class", seq_len(G))
  random_effect_names <- unlist(lapply(data$outcome_names, function(outcome) {
    paste0(outcome, c(":intercept", ":time"))
  }), use.names = FALSE)
  for (g in seq_len(G)) {
    names(beta[[g]]) <- data$coefficient_names[[g]]
    dimnames(Omega[[g]]) <- list(random_effect_names, random_effect_names)
    dimnames(Sigma[[g]]) <- list(data$outcome_names, data$outcome_names)
  }
  names(beta) <- names(Omega) <- names(Sigma) <- class_names
  colnames(xi) <- c("(Intercept)", data$gating_names)
  if (G > 1L) rownames(xi) <- class_names[seq_len(G - 1L)]
  rownames(B) <- as.character(data$subject_id)
  colnames(B) <- class_names
  class_assignment <- max.col(tau, ties.method = "first")
  names(class_assignment) <- as.character(data$subject_id)
  rownames(tau) <- as.character(data$subject_id)
  colnames(tau) <- class_names
  rownames(pi) <- as.character(data$subject_id)
  colnames(pi) <- class_names

  variational_fields <- names(variational_conditional)
  for (field in variational_fields) {
    names(variational_conditional[[field]]) <- class_names
  }
  names(variational$m_joint) <- names(variational$C_joint) <- class_names

  fit <- list(
    beta = beta,
    Omega = Omega,
    Sigma = Sigma,
    gating_coefficients = xi,
    prior_class_probabilities = pi,
    posterior_class_probabilities = tau,
    class_assignment = class_assignment,
    class_evidence = B,
    n_parameters = parameter_count$total,
    parameter_count = parameter_count$breakdown,
    variational_BIC = unname(variational_BIC),
    final_ELBO = unname(final_elbo),
    ELBO_trace = elbo_trace,
    converged = converged,
    iterations = completed_iterations,
    elapsed_seconds = unname(timing["elapsed"]),
    initial_clusters = population$clusters,
    gating_fit = gating_fit,
    variational = list(
      m_joint = variational$m_joint,
      C_joint = variational$C_joint,
      mu_b = variational_conditional$mu_b_list,
      S_b = variational_conditional$S_b_list,
      A = variational_conditional$A_list,
      L = variational_conditional$L_list,
      V = variational_conditional$V_list,
      m_eta = variational_conditional$m_eta_list,
      C_etaeta = variational_conditional$C_etaeta_list
    ),
    model = list(
      G = G,
      n = n,
      p = p,
      q_beta = data$q_beta,
      q_gating = qc,
      random_effects = "outcome-specific random intercept and random time slope",
      offset_included = any(vapply(data$offset_list, function(x) any(x != 0), logical(1)))
    ),
    control = list(
      max_iter = max_iter,
      inner_cycles = inner_cycles,
      tol = tol,
      convergence = convergence,
      tol_inner = tol_inner,
      gating_decay = gating_decay,
      init_method = init_method,
      init_seed = init_seed
    ),
    data = data
  )
  class(fit) <- "mpln_vem_fit"
  fit
}

fit_mpln_vem <- function(data,
                         n_starts = 1L,
                         start_seeds = NULL,
                         return_all_starts = FALSE,
                         ...) {
  if (length(n_starts) != 1L || !is.finite(n_starts) || n_starts < 1L ||
      n_starts != as.integer(n_starts)) {
    stop("n_starts must be a positive integer.", call. = FALSE)
  }
  n_starts <- as.integer(n_starts)
  dots <- list(...)
  if ("init_seed" %in% names(dots)) {
    stop("Specify initialization seeds with start_seeds, not init_seed.",
         call. = FALSE)
  }
  if (is.null(start_seeds)) start_seeds <- seq_len(n_starts)
  if (length(start_seeds) != n_starts || any(!is.finite(start_seeds)) ||
      any(start_seeds < 0) || any(start_seeds != as.integer(start_seeds))) {
    stop("start_seeds must contain n_starts nonnegative integers.", call. = FALSE)
  }
  start_seeds <- as.integer(start_seeds)

  fits <- vector("list", n_starts)
  for (s in seq_len(n_starts)) {
    fits[[s]] <- do.call(
      fit_mpln_vem_once,
      c(list(data = data, init_seed = start_seeds[s]), dots)
    )
  }
  final_elbo <- vapply(fits, function(x) x$final_ELBO, numeric(1))
  best <- which.max(final_elbo)
  fit <- fits[[best]]
  fit$selected_start <- best
  fit$start_summary <- data.frame(
    start = seq_len(n_starts),
    seed = start_seeds,
    final_ELBO = final_elbo,
    converged = vapply(fits, function(x) x$converged, logical(1)),
    iterations = vapply(fits, function(x) x$iterations, integer(1)),
    elapsed_seconds = vapply(fits, function(x) x$elapsed_seconds, numeric(1))
  )
  if (return_all_starts) fit$all_starts <- fits
  fit
}

print.mpln_vem_fit <- function(x, ...) {
  cat("Latent-class Poisson-lognormal mixed-model fit\n")
  cat("  Subjects:", x$model$n, "\n")
  cat("  Outcomes:", x$model$p, "\n")
  cat("  Classes:", x$model$G, "\n")
  cat("  Free parameters:", x$n_parameters, "\n")
  cat("  Final ELBO:", format(x$final_ELBO, digits = 8), "\n")
  cat("  Variational BIC:", format(x$variational_BIC, digits = 8), "\n")
  cat("  Converged:", if (x$converged) "yes" else "no", "\n")
  cat("  Iterations:", x$iterations, "\n")
  invisible(x)
}
