# Latent-Class Multivariate Poisson–Lognormal Mixed Models

Fit latent-class multivariate Poisson–lognormal (MPLN) mixed models to longitudinal count data using variational expectation–maximization (VEM). The R interface handles data preparation, initialization, and results. An RcppArmadillo source code performs the main numerical updates.

The model combines latent subject classes, outcome-specific random intercepts and time slopes, and correlated visit-level Gaussian noise. It supports unequal numbers of visits, irregular observation times, class-specific fixed-effect designs, and covariates that explain class membership.

This repository contains sourceable R scripts and a C++ core. It is **not an installable R package**: source the R interface and compile the core in each new R session before fitting.

## Contents

- [Repository structure](#repository-structure)
- [Requirements and setup](#requirements-and-setup)
- [Quick start](#quick-start)
- [Included examples](#included-examples)
- [Model specification](#model-specification)
- [Preparing your data](#preparing-your-data)
- [Fitting and initialization](#fitting-and-initialization)
- [Working with results](#working-with-results)
- [Convergence and model comparison](#convergence-and-model-comparison)
- [Smoke test](#smoke-test)
- [Troubleshooting and implementation limits](#troubleshooting-and-implementation-limits)

## Repository structure

```text
.
├── README.md
├── R/
│   └── mpln_vem.R               # Data preparation, fitting, and R interface
├── src/
│   └── mpln_vem_core.cpp        # RcppArmadillo numerical core
├── examples/
│   ├── example_G1_n500.R        # One class, 500 subjects, 3 outcomes
│   ├── example_G2_n700.R        # Two classes, 700 subjects, 3 outcomes
│   └── example_G3_n500.R        # Three classes, 500 subjects, 2 outcomes
└── tests/
    └── smoke_test.R            # Small integration checks for G = 1 and G = 2
```

## Requirements and setup

You need R, an R-compatible C++ build toolchain, and these R packages:

| Package | Purpose |
| --- | --- |
| `Rcpp` | Compile and load the C++ interface through `sourceCpp()` |
| `RcppArmadillo` | C++ linear algebra |
| `MASS` | Required by the interface; used for Gaussian simulation in the examples |
| `nnet` | Multinomial logistic class membership model |
| `mclust` | Gaussian mixture initialization |

Install the required packages in R:

```r
install.packages(c("Rcpp", "RcppArmadillo", "MASS", "nnet"))

# Optional, for init_method = "mclust":
install.packages("mclust")
```

Compiling the core requires build tools even when the R packages were installed as binaries. Use the build tools appropriate to your R installation: Rtools on Windows, Xcode Command Line Tools on macOS, or the R development tools and a C++ compiler on Linux. The compiler must support the requirements of your installed `RcppArmadillo` version.

Download or clone the repository, then **set your working directory to its root**, where `R/`, `src/`, `examples/`, and `tests/` are located. All commands below assume this working directory.

```r
source("R/mpln_vem.R")
compile_mpln_vem()
```

`compile_mpln_vem()` defaults to `src/mpln_vem_core.cpp`. To see compiler output or force recompilation:

```r
compile_mpln_vem(rebuild = TRUE, verbose = TRUE)
```

To check your installation before running a larger example, run this from a terminal at the repository root:

```sh
Rscript tests/smoke_test.R
```

Successful execution ends with `All smoke tests passed.`

## Quick start

This self-contained example simulates a small dataset with two outcomes and fits a single-class model. Its modest iteration limit is intended for learning the interface. Check convergence before interpreting estimates.

Run the following in R from the repository root:

```r
source("R/mpln_vem.R")
compile_mpln_vem()

set.seed(123)
n <- 30L
p <- 2L
time <- replicate(n, c(-1, -0.3, 0.3, 1), simplify = FALSE)
X <- lapply(time, function(t) cbind("(Intercept)" = 1, time = t))

Y <- lapply(time, function(t) {
  b <- rnorm(p, sd = 0.25)
  eta <- rbind(
    1.2 + 0.3 * t + b[1],
    1.0 - 0.2 * t + b[2]
  )
  eta <- eta + matrix(rnorm(length(eta), sd = 0.2), nrow = p)
  counts <- matrix(rpois(length(eta), lambda = exp(eta)), nrow = p)
  rownames(counts) <- c("outcome1", "outcome2")
  counts
})

data <- make_mpln_data(
  Y = Y,
  time = time,
  X = X,
  G = 1L,
  subject_id = paste0("subject", seq_len(n))
)

fit <- fit_mpln_vem(
  data,
  n_starts = 1L,
  start_seeds = 101L,
  max_iter = 100L,
  verbose = FALSE
)

print(fit)
fit$beta[[1]]
fit$Omega[[1]]
fit$Sigma[[1]]
plot(fit$ELBO_trace, type = "l", xlab = "Iteration", ylab = "ELBO")
```

The main workflow is always the same:

1. `compile_mpln_vem()` loads the numerical core.
2. `make_mpln_data()` validates and organizes the inputs.
3. `fit_mpln_vem()` fits the model and returns an `mpln_vem_fit` object.

## Included examples

The three scripts simulate data, construct the model inputs, fit the model, and inspect estimates. Each script sources the R interface and compiles the core automatically.

| Script | Classes ($G$) | Subjects ($n$) | Outcomes ($p$) | Visits per subject ($T_i$) | Starts | Free parameters |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| [example_G1_n500.R](examples/example_G1_n500.R) | 1 | 500 | 3 | 4–6 | 1 | 24 |
| [example_G2_n700.R](examples/example_G2_n700.R) | 2 | 700 | 3 | 3–7 | 3 | 51 |
| [example_G3_n500.R](examples/example_G3_n500.R) | 3 | 500 | 2 | 2–5 | 20 | 55 |

Run one script from a terminal:

```sh
Rscript examples/example_G1_n500.R
Rscript examples/example_G2_n700.R
Rscript examples/example_G3_n500.R
```

Alternatively, source one script in an interactive R session to retain its objects:

```r
source("examples/example_G2_n700.R")

fit$start_summary
head(fit$posterior_class_probabilities)
table(fitted = fit$class_assignment, truth = example$truth$class)
```

Each script creates `example$data`, `example$truth`, and `fit`. Sourcing another example in the same environment replaces those objects. The scripts do not save fitted objects automatically.

All three examples allow up to **2,000 iterations per start**. Starts run sequentially, and the three-class example fits 20 starts, so these are larger simulation examples rather than quick installation checks. Their fixed seeds make the simulation and initialization reproducible within an environment. Numerical results can still vary with package versions and computing platforms.

## Model specification

Let subject $i$ belong to latent class $Z_i \in \{1,\ldots,G\}$. At visit $l$, outcome $j$ is a count with conditional distribution

```math
Y_{ijl} \mid \eta_{ijlg}, Z_i = g \sim \text{Poisson}\!\left(\exp(\eta_{ijlg})\right).
```

For class $g$, the log-intensity vector $\boldsymbol\eta_{ilg}$ at a visit is

```math
\boldsymbol\eta_{ilg}
= o_{il}\mathbf{1}_p + X_{ilg}\boldsymbol\beta_g
+ M_{il}\mathbf{b}_{ig} + \boldsymbol\epsilon_{ilg},
\qquad
\mathbf{b}_{ig} \sim N(0,\Omega_g),
\qquad
\boldsymbol\epsilon_{ilg} \sim N(0,\Sigma_g).
```

Here, $X_{ilg}$ is the outcome-expanded fixed-effect design and $M_{il}$ places a random intercept and random time slope on each outcome.

- **Fixed effects:** `beta[[g]]` is class-specific. Classes may share a design while estimating different coefficients, or use different designs and coefficient dimensions.
- **Subject random effects:** `Omega[[g]]` is a $2p \times 2p$ covariance matrix, constrained to be block diagonal across outcomes. Each outcome has a $2 \times 2$ random-intercept/slope covariance block.
- **Visit-level variation:** `Sigma[[g]]` is an unrestricted $p \times p$ covariance matrix. Gaussian visit errors are independent across visits, with covariance $I_{T_i} \otimes \Sigma_g$ across the stacked visits. Shared subject random effects induce dependence over time.
- **Offsets:** $o_{il}$ is a known additive log-scale offset, shared by all outcomes at a visit.

For `G > 1`, class membership probabilities follow a multinomial logistic model based on subject-level covariates $\mathbf{c}_i$:

```math
\log\frac{\pi_{ig}}{\pi_{iG}}
= \xi_{g0} + \mathbf{c}_i^\top\boldsymbol\xi_g,
\qquad g=1,\ldots,G-1.
```

Class `G` is the reference class. With `gating = NULL`, the gating model is intercept-only. For `G = 1`, the gating model is bypassed and all class probabilities are one.

VEM uses a joint Gaussian variational approximation to each subject's random effects and latent log-intensities within each class. The fitting loop updates these distributions, class responsibilities, model parameters (including gating coefficients), then evaluates the evidence lower bound (ELBO).

## Preparing your data

```r
data <- make_mpln_data(
  Y = Y,
  time = time,
  X = X,
  G = 2L,
  gating = NULL,
  offset = NULL,
  subject_id = NULL
)
```

Use the same subject order in every input. Let `n` be the number of subjects, `p` the number of outcomes, and `T_i` the number of recorded visits for subject `i`.

| Argument | Required format |
| --- | --- |
| `Y` | List of length `n`; `Y[[i]]` is a numeric `p × T_i` matrix, with outcomes in rows and visits in columns. Counts must be finite, nonnegative integers without missing values. |
| `time` | List of length `n`; `time[[i]]` contains `T_i` finite numeric visit times aligned with the columns of `Y[[i]]`. |
| `X` | A list of `n` fixed-effect matrices shared across classes, or a list of `G` such subject lists. See the design options below. |
| `G` | Positive integer number of classes. Use `G <= n`. |
| `gating` | Optional finite numeric `n × q_c` matrix of subject-level covariates. The intercept is added internally; do not include an intercept column. `NULL` gives an intercept-only gating model. |
| `offset` | `NULL` for zero offsets, one finite scalar for all subjects and visits, or a list of length `n` whose entries are scalars or vectors of length `T_i`. Values are on the log scale. |
| `subject_id` | Optional vector of `n` unique subject identifiers; defaults to `seq_len(n)`. Inputs are matched by position, not joined by these identifiers. |

All subjects must have the same outcomes in the same row order. Each recorded visit must contain every outcome: partial outcome missingness is not supported. Omit entirely unobserved visits from `Y`, `time`, `X`, and visit-specific offsets together. Every subject must retain at least one recorded visit. Irregular times and different visit counts across subjects are allowed.

### Fixed-effect design options

**Standard design: one row per visit.** Supply a `T_i × q` matrix for each subject. The interface expands this into separate coefficients for each outcome. Include an intercept column explicitly if you want a fixed intercept.

```r
X <- lapply(time, function(t) cbind("(Intercept)" = 1, time = t))
```

With two outcomes, this design produces coefficients in the order `outcome1:(Intercept)`, `outcome1:time`, `outcome2:(Intercept)`, `outcome2:time`. Sharing this design across classes does **not** constrain coefficients to be equal across classes.

**Class-specific designs.** Supply one subject-level list per class. For example, class 1 can have linear time effects and class 2 quadratic time effects:

```r
X_by_class <- list(
  lapply(time, function(t) cbind("(Intercept)" = 1, time = t)),
  lapply(time, function(t) cbind("(Intercept)" = 1, time = t, time2 = t^2))
)

data_two_classes <- make_mpln_data(
  Y = Y, time = time, X = X_by_class, G = 2L
)
```

Supply a design for **every subject in every class**, since class membership is unknown. The number and order of columns must agree across subjects within each class. Different classes may use different numbers of columns.

**Custom expanded designs.** Supply a `p*T_i × d_g` matrix directly to control how coefficients are shared across outcomes. These matrices are used without automatic expansion. Rows must follow visit-major order:

```text
visit 1: outcome 1, outcome 2, ..., outcome p
visit 2: outcome 1, outcome 2, ..., outcome p
...
```

This is the order produced by `as.vector(Y[[i]])` in R. All design entries should be finite numeric values; encode categorical predictors before passing them. Use consistent column names and avoid redundant columns.

### Class membership covariates and exposure offsets

The following illustrates the structure for subject-level covariates and known, positive exposure durations. Replace the simulated values with your own data:

```r
set.seed(123)
gating <- cbind(baseline_score = rnorm(length(Y)))
exposure <- lapply(time, function(t) runif(length(t), 0.8, 1.2))
offset <- lapply(exposure, log)

data_with_covariates <- make_mpln_data(
  Y = Y,
  time = time,
  X = X,
  G = 2L,
  gating = gating,
  offset = offset
)
```

Pass `log(exposure)`, not raw exposure. Offsets are known quantities and add no fitted parameters. Outcome-specific offsets within the same visit are not supported.

The `time` input controls the random slope. Including time in the fixed-effect design is a separate choice. Use a centering and scaling convention suitable for your own data.

## Fitting and initialization

For a multi-class dataset, use several initializations and inspect the results across starts:

```r
fit <- fit_mpln_vem(
  data_with_covariates,
  n_starts = 3L,
  start_seeds = 101:103,
  init_method = "kmeans",
  max_iter = 500L,
  tol = 1e-4,
  convergence = "absolute",
  verbose = FALSE
)

fit$start_summary
fit$selected_start
```

The returned fit is the start with the **largest final ELBO**, even if that start has not met the convergence criterion. Errors are not skipped automatically: an error in a start interrupts the fitting call.

### Main controls

Pass these arguments directly to `fit_mpln_vem()`; there is no `control = list(...)` argument.

| Argument | Default | Meaning |
| --- | --- | --- |
| `n_starts` | `1L` | Number of sequential fits |
| `start_seeds` | `seq_len(n_starts)` | One nonnegative integer seed per start; use this argument rather than `init_seed` |
| `return_all_starts` | `FALSE` | Retain every complete fit in `fit$all_starts` |
| `init_method` | `"kmeans"` | `"kmeans"`, `"mclust"`, or `"random"` |
| `max_iter` | `500L` | Maximum outer VEM iterations per start |
| `tol` | `1e-4` | Stopping threshold for the ELBO change |
| `convergence` | `"absolute"` | `"absolute"` or `"relative"` ELBO change |
| `inner_cycles` | `10L` | Maximum mean/covariance update cycles per subject and class in each V-step |
| `tol_inner` | `1e-4` | Inner variational update tolerance |
| `verbose` | `TRUE` | Print the ELBO and its change at each outer iteration |

K-means and optional `mclust` initialization use standardized subject summaries of offset-adjusted log-counts, their variability and time trends, plus any gating covariates. If `mclust` is unavailable or its initialization fails, the code falls back to k-means. `G = 1` does not require clustering.

### User-supplied starting values

These optional arguments are also passed directly to `fit_mpln_vem()`:

| Argument | Format |
| --- | --- |
| `beta_init` | List of `G` coefficient vectors of lengths `data$q_beta[g]`; alternatively a `q_beta × G` matrix when all classes have the same coefficient dimension |
| `Omega_init` | List of `G` matrices, each `2p × 2p` |
| `Sigma_init` | List of `G` matrices, each `p × p` |
| `xi_init` | `(G-1) × (q_c+1)` gating coefficient matrix; intercept first, followed by covariates in input order |
| `tau_init` | `n × G` nonnegative finite class weights with positive row sums; rows are normalized internally |

Covariance starting values are symmetrized and given positive eigenvalue floors. For `Omega_init`, off-block entries across outcomes are discarded to enforce the model structure. Supplied starting values are reused across starts, so changing seeds may not produce distinct fits when initialization is fully specified. If no specified initialization, default initialization will be used.

<details>
<summary>Advanced numerical controls</summary>

| Argument | Default | Meaning |
| --- | --- | --- |
| `max_newton` | `1L` | Maximum Newton steps per inner mean update |
| `max_fixed` | `1L` | Maximum fixed-point steps per inner covariance update |
| `max_backtrack` | `10L` | Maximum backtracking attempts per inner update |
| `diag_perturb` | `0.25` | Diagonal perturbation used to stabilize Newton systems |
| `rcond_tol` | `0` | Conditioning threshold; a nonpositive value uses the square root of machine epsilon |
| `exp_cap` | `50` | Bounds exponential arguments to limit numerical overflow |
| `gating_maxit` | `200L` | Gating optimizer iteration limit; capped at 25 for outer iterations 1–10 and 75 for iterations 11–30 |
| `gating_reltol` | `1e-8` | Relative tolerance of the gating optimizer |
| `gating_decay` | `1e-4` | Weight decay passed to `nnet::multinom()` |
| `weight_floor` | `1e-8` | Threshold for retaining current class parameters when total responsibility is too small |

See [the R interface](R/mpln_vem.R) for the full implementation and argument validation.

</details>

## Working with results

`print(fit)` reports model dimensions, the free parameter count, final ELBO, variational BIC, convergence status, and iterations for the selected start.

| Field | Contents |
| --- | --- |
| `beta` | Named list of class-specific fixed-effect coefficient vectors |
| `Omega` | Named list of class-specific random-effect covariance matrices |
| `Sigma` | Named list of class-specific visit-level covariance matrices |
| `gating_coefficients` | `(G-1) × (q_c+1)` coefficients relative to class `G`; zero rows for `G = 1` |
| `prior_class_probabilities` | `n × G` probabilities from the fitted gating model |
| `posterior_class_probabilities` | `n × G` variational class responsibilities incorporating the observed outcomes |
| `class_assignment` | Highest-posterior-probability class per subject; ties choose the first class |
| `class_evidence` | `n × G` class-specific variational lower-bound contributions |
| `final_ELBO`, `ELBO_trace` | Final objective and its iteration history |
| `variational_BIC` | ELBO-based model comparison criterion |
| `n_parameters`, `parameter_count` | Total free parameter count and its breakdown |
| `converged`, `iterations` | Stopping status and outer iterations completed |
| `start_summary`, `selected_start` | Per-start results and index of the selected start |
| `elapsed_seconds` | Time spent in the selected start's VEM loop, excluding compilation, initialization, and result extraction |
| `model`, `control`, `data` | Model metadata, selected control settings, and prepared inputs |
| `variational` | Subject- and class-specific approximate Gaussian variational distributions |

For example:

```r
fit$beta[[1]]
fit$gating_coefficients
head(fit$posterior_class_probabilities)
table(fit$class_assignment)
colSums(fit$posterior_class_probabilities)  # Effective class sizes

assignments <- data.frame(
  subject_id = fit$data$subject_id,
  class = unname(fit$class_assignment),
  max_probability = apply(fit$posterior_class_probabilities, 1, max)
)
head(assignments)

# Optional: save the fitted object and record the software environment.
saveRDS(fit, "mpln_fit.rds")
sessionInfo()
```

Class labels are arbitrary. Match classes before comparing fitted parameters with simulation truth or with another fit.

### Variational distributions

Access these fields as `fit$variational$field[[g]][[i]]`, using class index `g` and subject position `i`:

- `m_joint`, `C_joint`: joint Gaussian mean and covariance of the stacked vector `(b, eta)`, of dimension `2p + p*T_i`.
- `mu_b`, `S_b`: marginal random-effect mean and covariance.
- `m_eta`, `C_etaeta`: marginal latent log-intensity mean and covariance, ordered by visit.
- `A`, `L`, `V`: conditional parameters defining `q(eta | b, class = g) = N(A + L b, V)`.

The random-effect order is outcome 1 intercept/slope, outcome 2 intercept/slope, and so on. These are approximate latent-variable distributions, not standard errors for the population parameters.

## Convergence and model comparison

The stopping rule compares successive ELBO values. Absolute convergence uses

```math
\left|\mathcal{L}_{k}-\mathcal{L}_{k-1}\right| < \texttt{tol},
```

and relative convergence divides this difference by $\max(1, |\mathcal{L}_{k-1}|)$. Reaching `max_iter` returns a fit with `converged = FALSE`; it does not automatically raise an error.

Inspect both the selected fit and the individual starts:

```r
fit$converged
fit$start_summary
plot(fit$ELBO_trace, type = "l", xlab = "Iteration", ylab = "ELBO")
range(diff(fit$ELBO_trace))
```

Convergence only indicates that the implemented stopping criterion was met. It does not establish a global optimum. Inner update limits, numerical safeguards, and the penalized gating update also mean that strict monotonicity of the reported ELBO should not be assumed.

The implementation reports

```math
\text{variational BIC} = -2\mathcal{L}_{\mathrm{final}} + k\log(n),
```

where `n` is the **number of subjects**, not the number of visits or count observations. For `d_g = data$q_beta[g]` and `q_c` gating covariates, the parameter count is

```math
k = \sum_{g=1}^{G} d_g + 3Gp
+ G\frac{p(p+1)}{2} + (G-1)(q_c+1).
```

The terms count fixed effects, outcome-specific random intercept/slope covariance blocks, visit-level covariance matrices, and gating coefficients. Known offsets and variational parameters are excluded.

Lower variational BIC is preferred when comparing candidate models fitted to the same observations with the same offset convention. This is an **ELBO-based approximation**, not BIC computed from the exact maximized marginal likelihood. To compare class counts, construct and fit a separate data object for each `G`, use multiple starts, and check convergence and effective class sizes alongside the criterion. The interface does not select `G` automatically.

## Smoke test

From the repository root:

```sh
Rscript tests/smoke_test.R
```

The test uses 18 simulated subjects, two outcomes, and three visits per subject. It checks:

- `G = 1`: unit class probabilities, an empty gating coefficient matrix, 13 free parameters, and finite variational BIC.
- `G = 2`: class-specific designs with coefficient dimensions 4 and 6, two gating covariates, nonzero offsets, normalized posterior probabilities, 31 free parameters, and finite variational BIC.

Each fit runs only three outer iterations. This is an integration check for compilation and the fitting workflow; it does not test statistical recovery or convergence, and it does not run the larger `G = 3` example.

## Troubleshooting and implementation limits

| Problem | What to check |
| --- | --- |
| Cannot find `R/mpln_vem.R` or the C++ source | Confirm `getwd()` points to the repository root. |
| Missing package error | Install the packages named in the error into a library visible to the current R session. |
| Compilation fails | Confirm the R-compatible build tools are installed; rerun `compile_mpln_vem(rebuild = TRUE, verbose = TRUE)` to inspect the compiler error. |
| “The Rcpp core is not loaded” | Run `compile_mpln_vem()` in the current session before fitting. |
| Invalid counts or missing values | Check that `Y` has outcomes in rows, contains integer counts, and has no partially observed visits. |
| Design or offset dimension error | Check each subject's `T_i`, use visit-major rows for expanded designs, and pass offsets as visit-level scalars or vectors. |
| K-means cannot form the requested clusters | Check `G`, the number of distinct subject summaries, and whether `init_method = "random"` is appropriate. |
| `converged = FALSE` or unstable estimates | Inspect the ELBO trace and class sizes; check covariate scaling and design redundancy, then consider more iterations or additional starts. |
| High memory use or long runtime | Begin with fewer subjects or starts to check the workflow. Dense variational covariance matrices grow with `2p + p*T_i` for every subject and class. |

The current implementation fixes the random-effect design to an intercept and a time slope for each outcome and enforces block-diagonal `Omega`. It does not provide arbitrary random-effect designs, partially missing outcome handling, or zero-inflation components. All starts are computed sequentially, and the wrapper holds their fitted objects during selection, even when `return_all_starts = FALSE`.
