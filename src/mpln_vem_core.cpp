// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <cmath>
#include <limits>
using namespace Rcpp;
using namespace arma;

// -----------------------------------------------------------------------------
// Variational source code for the latent-class Poisson-lognormal mixed model.
//
// This implementation supports any G >= 1. 
// The fitted model imposes:
//   (1) Omega_g block-diagonal across outcomes;
//   (2) K_{Sigma,ig} = I_{T_i} \otimes Sigma_g, i.e. latent Gaussian noise
//       is independent across visit times within each class.
//
// Xbig_by_class is a list of length G; element g is a subject-level list of
// fixed-effect design matrices. This permits different fixed-effect designs
// and coefficient dimensions across classes. offset_list contains known
// offsets and contributes no fitted parameter. Every outcome is
// observed at each recorded visit; visits with no observed outcomes are absent.
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Utility functions
// -----------------------------------------------------------------------------
inline arma::mat symm(const arma::mat& A){ return 0.5 * (A + A.t()); }

inline arma::mat spd_floor(const arma::mat& A, double floor_val = 1e-8){
  arma::mat B = symm(A);
  arma::vec eigval;
  arma::mat eigvec;
  if(!arma::eig_sym(eigval, eigvec, B)){
    return B + floor_val * arma::eye(B.n_rows, B.n_cols);
  }
  for(arma::uword k = 0; k < eigval.n_elem; ++k){
    if(!std::isfinite(eigval(k)) || eigval(k) < floor_val) eigval(k) = floor_val;
  }
  return symm(eigvec * arma::diagmat(eigval) * eigvec.t());
}

inline arma::mat blockdiag_spd_floor(const arma::mat& A,
                                     int p,
                                     int r,
                                     double floor_val = 1e-8){
  int db = p * r;
  arma::mat out(db, db, arma::fill::zeros);
  for(int j = 0; j < p; ++j){
    arma::uword a = j * r;
    arma::uword b = (j + 1) * r - 1;
    arma::mat block = A.submat(a, a, b, b);
    block = spd_floor(block, floor_val);
    out.submat(a, a, b, b) = block;
  }
  return symm(out);
}

inline arma::mat safe_inv_sympd(const arma::mat& A, double jitter = 1e-8){
  arma::mat B = symm(A), out;
  double eps = jitter;
  for(int k = 0; k < 12; ++k){
    if(arma::inv_sympd(out, B + eps * arma::eye(B.n_rows, B.n_cols))) return out;
    eps *= 10.0;
  }
  B = spd_floor(B, eps);
  if(arma::inv_sympd(out, B)) return out;
  Rcpp::stop("safe_inv_sympd failed");
}

inline double safe_logdet_sympd(const arma::mat& A, double jitter = 1e-8){
  arma::mat B = symm(A);
  double val = 0.0, sign = 0.0;
  double eps = jitter;
  for(int k = 0; k < 12; ++k){
    arma::log_det(val, sign, B + eps * arma::eye(B.n_rows, B.n_cols));
    if(std::isfinite(val) && sign > 0) return val;
    eps *= 10.0;
  }
  B = spd_floor(B, eps);
  arma::log_det(val, sign, B);
  if(std::isfinite(val) && sign > 0) return val;
  Rcpp::stop("safe_logdet_sympd failed");
}

inline double robust_rcond(const arma::mat& A){
  double rc = arma::rcond(A);
  return std::isfinite(rc) ? rc : 0.0;
}

inline arma::mat stabilize_hessian(const arma::mat& H,
                                   double rcond_tol,
                                   double diag_perturb){
  arma::mat Hs = symm(H);
  if(robust_rcond(Hs) >= rcond_tol) return Hs;

  arma::vec d = Hs.diag();
  arma::mat Hd = arma::diagmat(d);
  if(robust_rcond(Hd) >= rcond_tol) return Hd;

  arma::vec absd = arma::abs(d);
  arma::vec pert = diag_perturb * arma::clamp(absd, 1e-8, arma::datum::inf);
  arma::vec dnew = d;
  for(arma::uword k = 0; k < dnew.n_elem; ++k){
    if(dnew(k) <= 0) dnew(k) -= pert(k);
    else dnew(k) = -pert(k);
  }
  return arma::diagmat(dnew);
}

inline double safe_exp_scalar(double x, double cap = 50.0){
  if(x > cap) x = cap;
  if(x < -cap) x = -cap;
  return std::exp(x);
}

inline arma::vec vectorise_by_visit(const arma::mat& A){
  // Armadillo vectorise stacks columns:
  // (outcome 1...p at visit 1, outcome 1...p at visit 2, ...).
  return arma::vectorise(A);
}

arma::mat build_M_big_from_list(const Rcpp::List& M_sub_r, int T_i, int p, int db){
  arma::mat Mbig(p * T_i, db, arma::fill::zeros);
  for(int l = 0; l < T_i; ++l){
    arma::mat Ml = Rcpp::as<arma::mat>(M_sub_r[l]);
    if((int)Ml.n_rows != p || (int)Ml.n_cols != db){
      Rcpp::stop("Each M_il must have dimension p x (p*r).");
    }
    Mbig.rows(l * p, (l + 1) * p - 1) = Ml;
  }
  return Mbig;
}

arma::mat build_prior_precision(const arma::mat& Mbig,
                                const arma::mat& Omega,
                                const arma::mat& Sigma,
                                int T_i){
  int db = Omega.n_rows;
  int dEta = Mbig.n_rows;
  arma::mat Omega_inv = safe_inv_sympd(Omega);
  arma::mat Sigma_inv = safe_inv_sympd(Sigma);
  arma::mat Kinv = arma::kron(arma::eye(T_i, T_i), Sigma_inv);

  arma::mat Q(db + dEta, db + dEta, arma::fill::zeros);
  Q.submat(0, 0, db - 1, db - 1) = Omega_inv + Mbig.t() * Kinv * Mbig;
  Q.submat(0, db, db - 1, db + dEta - 1) = -Mbig.t() * Kinv;
  Q.submat(db, 0, db + dEta - 1, db - 1) = -Kinv * Mbig;
  Q.submat(db, db, db + dEta - 1, db + dEta - 1) = Kinv;
  return symm(Q);
}

arma::vec compute_lambda_joint(const arma::vec& m,
                               const arma::mat& C,
                               int db,
                               double exp_cap){
  int dEta = m.n_elem - db;
  arma::vec lambda(dEta, arma::fill::zeros);
  arma::vec meta = m.subvec(db, db + dEta - 1);
  arma::vec diag_eta = C.submat(db, db, db + dEta - 1, db + dEta - 1).diag();
  for(int k = 0; k < dEta; ++k){
    lambda(k) = safe_exp_scalar(meta(k) + 0.5 * diag_eta(k), exp_cap);
  }
  return lambda;
}

// -----------------------------------------------------------------------------
// Complete class-specific lower bound B_ig.
// -----------------------------------------------------------------------------
double subject_elbo_joint(const arma::mat& Y,
                          const arma::mat& Xbig,
                          const arma::mat& Mbig,
                          const arma::vec& offset,
                          const arma::vec& beta,
                          const arma::mat& Omega,
                          const arma::mat& Sigma,
                          const arma::vec& m,
                          const arma::mat& C,
                          int p,
                          int r,
                          double exp_cap){
  int T_i = Y.n_cols;
  int db = p * r;
  int dEta = p * T_i;
  int dAll = db + dEta;
  const double log2pi = std::log(2.0 * M_PI);

  if((int)offset.n_elem != dEta){
    Rcpp::stop("Each offset vector must have length p*T_i.");
  }
  arma::vec x_i = offset + Xbig * beta;
  arma::vec y_vec = vectorise_by_visit(Y);

  arma::vec mb = m.subvec(0, db - 1);
  arma::vec meta = m.subvec(db, db + dEta - 1);

  arma::mat Cbb = C.submat(0, 0, db - 1, db - 1);
  arma::mat Cee = C.submat(db, db, db + dEta - 1, db + dEta - 1);
  arma::mat Ceb = C.submat(db, 0, db + dEta - 1, db - 1);
  arma::mat Cbe = Ceb.t();

  arma::mat Omega_inv = safe_inv_sympd(Omega);
  arma::mat Sigma_inv = safe_inv_sympd(Sigma);
  arma::mat Kinv = arma::kron(arma::eye(T_i, T_i), Sigma_inv);

  // Psi_ig = E_q[(eta - X beta - M b)(eta - X beta - M b)^T].
  arma::vec mean_resid = meta - x_i - Mbig * mb;
  arma::mat Psi = Cee + Mbig * Cbb * Mbig.t() - Ceb * Mbig.t() - Mbig * Cbe
                  + mean_resid * mean_resid.t();
  Psi = symm(Psi);

  double val = 0.0;

  // E_q log p(b_ig | Z_i=g).
  val += -0.5 * db * log2pi;
  val += -0.5 * safe_logdet_sympd(Omega);
  val += -0.5 * arma::trace(Omega_inv * (Cbb + mb * mb.t()));

  // E_q log p(eta_ig | b_ig, Z_i=g).
  val += -0.5 * dEta * log2pi;
  val += -0.5 * T_i * safe_logdet_sympd(Sigma);
  val += -0.5 * arma::trace(Kinv * Psi);

  // E_q log p(Y_i | eta_ig, Z_i=g), including -log(Y_ijl!).
  arma::vec lambda = compute_lambda_joint(m, C, db, exp_cap);
  for(int k = 0; k < dEta; ++k){
    val += y_vec(k) * meta(k) - lambda(k) - std::lgamma(y_vec(k) + 1.0);
  }

  // Entropy of N(m_ig, C_ig).
  val += 0.5 * safe_logdet_sympd(C);
  val += 0.5 * dAll * (1.0 + log2pi);

  return val;
}

// -----------------------------------------------------------------------------
// Newton update for m_ig with line search.
// -----------------------------------------------------------------------------
arma::vec update_m_joint_newton_linesearch(const arma::mat& Y,
                                           const arma::mat& Xbig,
                                           const arma::mat& Mbig,
                                           const arma::vec& offset,
                                           const arma::vec& beta,
                                           const arma::mat& Omega,
                                           const arma::mat& Sigma,
                                           const arma::mat& Q,
                                           const arma::vec& m0,
                                           const arma::vec& m_old,
                                           const arma::mat& C,
                                           int p,
                                           int r,
                                           int max_newton,
                                           int max_backtrack,
                                           double tol,
                                           double rcond_tol,
                                           double diag_perturb,
                                           double exp_cap){
  int db = p * r;
  int dEta = p * Y.n_cols;
  int dAll = db + dEta;
  arma::vec y_vec = vectorise_by_visit(Y);
  arma::vec m = m_old;

  for(int iter = 0; iter < max_newton; ++iter){
    arma::vec lambda = compute_lambda_joint(m, C, db, exp_cap);
    arma::vec s(dAll, arma::fill::zeros);
    s.subvec(db, dAll - 1) = y_vec - lambda;

    arma::vec grad = -Q * (m - m0) + s;
    arma::mat D(dAll, dAll, arma::fill::zeros);
    D.submat(db, db, dAll - 1, dAll - 1) = arma::diagmat(lambda);
    arma::mat H = -Q - D;
    arma::mat Huse = stabilize_hessian(H, rcond_tol, diag_perturb);

    arma::vec step;
    bool ok = arma::solve(step, Huse, grad);
    if(!ok || !step.is_finite()) break;

    double elbo_cur = subject_elbo_joint(Y, Xbig, Mbig, offset, beta, Omega, Sigma,
                                         m, C, p, r, exp_cap);
    bool accepted = false;
    double alpha = 1.0;
    arma::vec m_new = m;

    for(int bt = 0; bt < max_backtrack; ++bt){
      m_new = m - alpha * step;
      double elbo_new = subject_elbo_joint(Y, Xbig, Mbig, offset, beta, Omega, Sigma,
                                           m_new, C, p, r, exp_cap);
      if(std::isfinite(elbo_new) && elbo_new >= elbo_cur - 1e-10){
        accepted = true;
        break;
      }
      alpha *= 0.5;
    }

    if(!accepted) break;
    if(arma::norm(m_new - m, 2) < tol){
      m = m_new;
      break;
    }
    m = m_new;
  }
  return m;
}

// -----------------------------------------------------------------------------
// Fixed-point covariance update for C_ig with damping.
// -----------------------------------------------------------------------------
arma::mat update_C_joint_fixedpoint_linesearch(const arma::mat& Y,
                                               const arma::mat& Xbig,
                                               const arma::mat& Mbig,
                                               const arma::vec& offset,
                                               const arma::vec& beta,
                                               const arma::mat& Omega,
                                               const arma::mat& Sigma,
                                               const arma::mat& Q,
                                               const arma::vec& m,
                                               const arma::mat& C_old,
                                               int p,
                                               int r,
                                               int max_fp,
                                               int max_backtrack,
                                               double tol,
                                               double exp_cap){
  int db = p * r;
  int dEta = p * Y.n_cols;
  int dAll = db + dEta;
  arma::mat C = symm(C_old);

  for(int it = 0; it < max_fp; ++it){
    arma::vec lambda = compute_lambda_joint(m, C, db, exp_cap);
    arma::mat Prec = Q;
    Prec.submat(db, db, dAll - 1, dAll - 1) += arma::diagmat(lambda);
    Prec = symm(Prec);

    arma::mat C_target = safe_inv_sympd(Prec);
    C_target = symm(C_target);

    double elbo_cur = subject_elbo_joint(Y, Xbig, Mbig, offset, beta, Omega, Sigma,
                                         m, C, p, r, exp_cap);
    bool accepted = false;
    double alpha = 1.0;
    arma::mat C_new = C;

    for(int bt = 0; bt < max_backtrack; ++bt){
      C_new = symm((1.0 - alpha) * C + alpha * C_target);
      double elbo_new = subject_elbo_joint(Y, Xbig, Mbig, offset, beta, Omega, Sigma,
                                           m, C_new, p, r, exp_cap);
      if(std::isfinite(elbo_new) && elbo_new >= elbo_cur - 1e-10){
        accepted = true;
        break;
      }
      alpha *= 0.5;
    }

    if(!accepted) break;
    if(arma::norm(C_new - C, "fro") < tol){
      C = C_new;
      break;
    }
    C = C_new;
  }
  return symm(C);
}

Rcpp::List convert_joint_to_conditional(const arma::vec& m,
                                        const arma::mat& C,
                                        int db,
                                        int dEta){
  arma::vec mu_b = m.subvec(0, db - 1);
  arma::vec m_eta = m.subvec(db, db + dEta - 1);
  arma::mat Cbb = symm(C.submat(0, 0, db - 1, db - 1));
  arma::mat Cee = symm(C.submat(db, db, db + dEta - 1, db + dEta - 1));
  arma::mat Ceb = C.submat(db, 0, db + dEta - 1, db - 1);
  arma::mat Cbe = Ceb.t();

  arma::mat Cbb_inv = safe_inv_sympd(Cbb);
  arma::mat L = Ceb * Cbb_inv;
  arma::vec A = m_eta - L * mu_b;
  arma::mat V = symm(Cee - Ceb * Cbb_inv * Cbe);
  V = spd_floor(V, 1e-10);

  return Rcpp::List::create(
    Rcpp::_["mu_b"] = mu_b,
    Rcpp::_["S_b"] = Cbb,
    Rcpp::_["A"] = A,
    Rcpp::_["L"] = L,
    Rcpp::_["V"] = V,
    Rcpp::_["m_eta"] = m_eta,
    Rcpp::_["C_etaeta"] = Cee
  );
}

// -----------------------------------------------------------------------------
// V-step for G >= 1 with flexible joint Gaussian variational familys.
// -----------------------------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List vstep_gmix_flexible_cpp(
    Rcpp::List Y_list,
    Rcpp::List Xbig_by_class,
    Rcpp::List M_list,
    Rcpp::List offset_list,
    Rcpp::List beta_list,
    Rcpp::List Omega_list,
    Rcpp::List Sigma_list,
    Rcpp::List m_joint_list,
    Rcpp::List C_joint_list,
    int p,
    int r,
    int inner_cycles = 10,
    int max_newton = 1,
    int max_backtrack = 10,
    int max_fixed = 1,
    double tol_inner = 1e-8,
    double rcond_tol = 0.0,
    double diag_perturb = 0.25,
    double exp_cap = 50.0){

  if(rcond_tol <= 0.0) rcond_tol = std::sqrt(std::numeric_limits<double>::epsilon());

  int n = Y_list.size();
  int G = beta_list.size();
  int db = p * r;

  if(Xbig_by_class.size() != G){
    Rcpp::stop("Xbig_by_class must be a list of length G.");
  }
  if(offset_list.size() != n){
    Rcpp::stop("offset_list must be a list of length n.");
  }

  arma::mat Bmat(n, G, arma::fill::zeros);
  Rcpp::List m_out(G), C_out(G);

  for(int g = 0; g < G; ++g){
    Rcpp::List Xbig_list = Xbig_by_class[g];
    if(Xbig_list.size() != n){
      Rcpp::stop("Each class-specific fixed-design list must have length n.");
    }
    arma::vec beta_g = Rcpp::as<arma::vec>(beta_list[g]);
    arma::mat Omega_g = blockdiag_spd_floor(Rcpp::as<arma::mat>(Omega_list[g]), p, r, 1e-8);
    arma::mat Sigma_g = spd_floor(Rcpp::as<arma::mat>(Sigma_list[g]), 1e-8);

    Rcpp::List m_g_in = m_joint_list[g];
    Rcpp::List C_g_in = C_joint_list[g];
    Rcpp::List m_g_out(n), C_g_out(n);

    for(int i = 0; i < n; ++i){
      arma::mat Y = Rcpp::as<arma::mat>(Y_list[i]);
      arma::mat Xbig = Rcpp::as<arma::mat>(Xbig_list[i]);
      arma::vec offset = Rcpp::as<arma::vec>(offset_list[i]);
      Rcpp::List M_sub_r = M_list[i];
      int T_i = Y.n_cols;
      int dEta = p * T_i;
      arma::mat Mbig = build_M_big_from_list(M_sub_r, T_i, p, db);
      if((int)Xbig.n_rows != dEta){
        Rcpp::stop("Each fixed-design matrix must have p*T_i rows.");
      }
      if((int)Xbig.n_cols != (int)beta_g.n_elem){
        Rcpp::stop("A class-specific fixed-design matrix has an incompatible number of columns.");
      }
      if((int)offset.n_elem != dEta){
        Rcpp::stop("Each offset vector must have length p*T_i.");
      }
      arma::vec x_i = offset + Xbig * beta_g;

      arma::mat Q = build_prior_precision(Mbig, Omega_g, Sigma_g, T_i);
      arma::vec m0(db + dEta, arma::fill::zeros);
      m0.subvec(db, db + dEta - 1) = x_i;

      arma::vec m = Rcpp::as<arma::vec>(m_g_in[i]);
      arma::mat C = Rcpp::as<arma::mat>(C_g_in[i]);
      C = symm(C);

      for(int cyc = 0; cyc < inner_cycles; ++cyc){
        arma::vec m_old = m;
        arma::mat C_old = C;

        m = update_m_joint_newton_linesearch(Y, Xbig, Mbig, offset, beta_g, Omega_g, Sigma_g,
                                             Q, m0, m, C, p, r,
                                             max_newton, max_backtrack,
                                             tol_inner, rcond_tol, diag_perturb, exp_cap);

        C = update_C_joint_fixedpoint_linesearch(Y, Xbig, Mbig, offset, beta_g, Omega_g, Sigma_g,
                                                 Q, m, C, p, r,
                                                 max_fixed, max_backtrack,
                                                 tol_inner, exp_cap);

        double diff = arma::norm(m - m_old, 2) + arma::norm(C - C_old, "fro");
        if(diff < tol_inner) break;
      }

      C = symm(C);
      Bmat(i, g) = subject_elbo_joint(Y, Xbig, Mbig, offset, beta_g, Omega_g, Sigma_g,
                                      m, C, p, r, exp_cap);

      m_g_out[i] = m;
      C_g_out[i] = C;
    }

    m_out[g] = m_g_out;
    C_out[g] = C_g_out;
  }

  return Rcpp::List::create(
    Rcpp::_["m_joint_list"] = m_out,
    Rcpp::_["C_joint_list"] = C_out,
    Rcpp::_["B"] = Bmat
  );
}

// -----------------------------------------------------------------------------
// Compute B = (B_ig) for fixed variational quantities and model parameters.
// -----------------------------------------------------------------------------
// [[Rcpp::export]]
arma::mat compute_B_gmix_flexible_cpp(
    Rcpp::List Y_list,
    Rcpp::List Xbig_by_class,
    Rcpp::List M_list,
    Rcpp::List offset_list,
    Rcpp::List beta_list,
    Rcpp::List Omega_list,
    Rcpp::List Sigma_list,
    Rcpp::List m_joint_list,
    Rcpp::List C_joint_list,
    int p,
    int r,
    double exp_cap = 50.0){

  int n = Y_list.size();
  int G = beta_list.size();
  int db = p * r;
  arma::mat Bmat(n, G, arma::fill::zeros);

  if(Xbig_by_class.size() != G){
    Rcpp::stop("Xbig_by_class must be a list of length G.");
  }
  if(offset_list.size() != n){
    Rcpp::stop("offset_list must be a list of length n.");
  }

  for(int g = 0; g < G; ++g){
    Rcpp::List Xbig_list = Xbig_by_class[g];
    if(Xbig_list.size() != n){
      Rcpp::stop("Each class-specific fixed-design list must have length n.");
    }
    arma::vec beta_g = Rcpp::as<arma::vec>(beta_list[g]);
    arma::mat Omega_g = blockdiag_spd_floor(Rcpp::as<arma::mat>(Omega_list[g]), p, r, 1e-8);
    arma::mat Sigma_g = spd_floor(Rcpp::as<arma::mat>(Sigma_list[g]), 1e-8);
    Rcpp::List m_g = m_joint_list[g];
    Rcpp::List C_g = C_joint_list[g];

    for(int i = 0; i < n; ++i){
      arma::mat Y = Rcpp::as<arma::mat>(Y_list[i]);
      arma::mat Xbig = Rcpp::as<arma::mat>(Xbig_list[i]);
      arma::vec offset = Rcpp::as<arma::vec>(offset_list[i]);
      Rcpp::List M_sub_r = M_list[i];
      int T_i = Y.n_cols;
      int dEta = p * T_i;
      arma::mat Mbig = build_M_big_from_list(M_sub_r, T_i, p, db);
      arma::vec m = Rcpp::as<arma::vec>(m_g[i]);
      arma::mat C = Rcpp::as<arma::mat>(C_g[i]);
      if((int)Xbig.n_rows != dEta || (int)Xbig.n_cols != (int)beta_g.n_elem){
        Rcpp::stop("A class-specific fixed-design matrix has incompatible dimensions.");
      }
      if((int)offset.n_elem != dEta){
        Rcpp::stop("Each offset vector must have length p*T_i.");
      }
      Bmat(i, g) = subject_elbo_joint(Y, Xbig, Mbig, offset, beta_g, Omega_g, Sigma_g,
                                      m, C, p, r, exp_cap);
    }
  }
  return Bmat;
}

// -----------------------------------------------------------------------------
// M-step for beta_g, Omega_g, Sigma_g using m_ig, C_ig and tau_ig.
// -----------------------------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List mstep_gmix_flexible_cpp(
    Rcpp::List Xbig_by_class,
    Rcpp::List M_list,
    Rcpp::List offset_list,
    arma::mat tau,
    Rcpp::List m_joint_list,
    Rcpp::List C_joint_list,
    Rcpp::List beta_list_current,
    Rcpp::List Omega_list_current,
    Rcpp::List Sigma_list_current,
    int p,
    int r,
    double weight_floor = 1e-8){

  int n = offset_list.size();
  int G = tau.n_cols;
  int db = p * r;

  if((int)tau.n_rows != n){
    Rcpp::stop("tau must have one row per subject.");
  }

  if(Xbig_by_class.size() != G){
    Rcpp::stop("Xbig_by_class must be a list of length G.");
  }

  Rcpp::List beta_out(G), Omega_out(G), Sigma_out(G);

  for(int g = 0; g < G; ++g){
    Rcpp::List Xbig_list = Xbig_by_class[g];
    if(Xbig_list.size() != n){
      Rcpp::stop("Each class-specific fixed-design list must have length n.");
    }
    arma::vec beta_cur = Rcpp::as<arma::vec>(beta_list_current[g]);
    arma::mat Omega_cur = blockdiag_spd_floor(Rcpp::as<arma::mat>(Omega_list_current[g]), p, r, 1e-8);
    arma::mat Sigma_cur = spd_floor(Rcpp::as<arma::mat>(Sigma_list_current[g]), 1e-8);
    Rcpp::List m_g = m_joint_list[g];
    Rcpp::List C_g = C_joint_list[g];

    double ng = arma::accu(tau.col(g));
    if(ng <= weight_floor){
      beta_out[g] = beta_cur;
      Omega_out[g] = Omega_cur;
      Sigma_out[g] = Sigma_cur;
      continue;
    }

    // beta_g update using eta_bar_ig = E_q(eta_ig - M_i b_ig).
    arma::mat Sigma_inv = safe_inv_sympd(Sigma_cur);
    int qtot = beta_cur.n_elem;
    arma::mat lhs(qtot, qtot, arma::fill::zeros);
    arma::vec rhs(qtot, arma::fill::zeros);

    for(int i = 0; i < n; ++i){
      double w = tau(i, g);
      if(w <= 0.0) continue;
      arma::mat Xbig = Rcpp::as<arma::mat>(Xbig_list[i]);
      arma::vec offset = Rcpp::as<arma::vec>(offset_list[i]);
      Rcpp::List M_sub_r = M_list[i];
      arma::vec m = Rcpp::as<arma::vec>(m_g[i]);
      if(Xbig.n_rows % p != 0 || (int)Xbig.n_cols != (int)beta_cur.n_elem){
        Rcpp::stop("A class-specific fixed-design matrix has incompatible dimensions.");
      }
      int T_i = Xbig.n_rows / p;
      int dEta = p * T_i;
      if((int)offset.n_elem != dEta){
        Rcpp::stop("Each offset vector must have length p*T_i.");
      }
      arma::mat Mbig = build_M_big_from_list(M_sub_r, T_i, p, db);
      arma::mat Kinv = arma::kron(arma::eye(T_i, T_i), Sigma_inv);
      arma::vec mb = m.subvec(0, db - 1);
      arma::vec meta = m.subvec(db, db + dEta - 1);
      arma::vec eta_bar = meta - offset - Mbig * mb;

      lhs += w * Xbig.t() * Kinv * Xbig;
      rhs += w * Xbig.t() * Kinv * eta_bar;
    }

    arma::vec beta_new = beta_cur;
    if(robust_rcond(lhs) > 1e-12){
      arma::vec tmp;
      bool ok = arma::solve(tmp, lhs, rhs);
      if(ok && tmp.is_finite()) beta_new = tmp;
    }

    // Omega_g update constrained to be block-diagonal across outcomes.
    arma::mat Omega_new(db, db, arma::fill::zeros);
    for(int j = 0; j < p; ++j){
      arma::uword a = j * r;
      arma::uword b = (j + 1) * r - 1;
      arma::mat block(r, r, arma::fill::zeros);
      for(int i = 0; i < n; ++i){
        double w = tau(i, g);
        if(w <= 0.0) continue;
        arma::vec m = Rcpp::as<arma::vec>(m_g[i]);
        arma::mat C = Rcpp::as<arma::mat>(C_g[i]);
        arma::vec mb = m.subvec(0, db - 1);
        arma::mat Cbb = C.submat(0, 0, db - 1, db - 1);
        arma::vec mbj = mb.subvec(a, b);
        arma::mat Cbbj = Cbb.submat(a, a, b, b);
        block += w * (Cbbj + mbj * mbj.t());
      }
      block = spd_floor(block / ng, 1e-8);
      Omega_new.submat(a, a, b, b) = block;
    }
    Omega_new = blockdiag_spd_floor(Omega_new, p, r, 1e-8);

    // Sigma_g update: use only diagonal visit blocks of Psi_ig.
    arma::mat Sigma_num(p, p, arma::fill::zeros);
    double denom = 0.0;
    for(int i = 0; i < n; ++i){
      double w = tau(i, g);
      if(w <= 0.0) continue;
      arma::mat Xbig = Rcpp::as<arma::mat>(Xbig_list[i]);
      arma::vec offset = Rcpp::as<arma::vec>(offset_list[i]);
      Rcpp::List M_sub_r = M_list[i];
      arma::vec m = Rcpp::as<arma::vec>(m_g[i]);
      arma::mat C = Rcpp::as<arma::mat>(C_g[i]);
      if(Xbig.n_rows % p != 0 || (int)Xbig.n_cols != (int)beta_new.n_elem){
        Rcpp::stop("A class-specific fixed-design matrix has incompatible dimensions.");
      }
      int T_i = Xbig.n_rows / p;
      int dEta = p * T_i;
      if((int)offset.n_elem != dEta){
        Rcpp::stop("Each offset vector must have length p*T_i.");
      }
      arma::mat Mbig = build_M_big_from_list(M_sub_r, T_i, p, db);
      arma::vec x_i = offset + Xbig * beta_new;

      arma::vec mb = m.subvec(0, db - 1);
      arma::vec meta = m.subvec(db, db + dEta - 1);
      arma::mat Cbb = C.submat(0, 0, db - 1, db - 1);
      arma::mat Cee = C.submat(db, db, db + dEta - 1, db + dEta - 1);
      arma::mat Ceb = C.submat(db, 0, db + dEta - 1, db - 1);
      arma::mat Cbe = Ceb.t();

      arma::vec mean_resid = meta - x_i - Mbig * mb;
      arma::mat Psi = Cee + Mbig * Cbb * Mbig.t() - Ceb * Mbig.t() - Mbig * Cbe
                      + mean_resid * mean_resid.t();
      Psi = symm(Psi);

      for(int l = 0; l < T_i; ++l){
        Sigma_num += w * Psi.submat(l * p, l * p, (l + 1) * p - 1, (l + 1) * p - 1);
        denom += w;
      }
    }

    arma::mat Sigma_new = Sigma_cur;
    if(denom > weight_floor){
      Sigma_new = spd_floor(Sigma_num / denom, 1e-8);
    }

    beta_out[g] = beta_new;
    Omega_out[g] = Omega_new;
    Sigma_out[g] = Sigma_new;
  }

  return Rcpp::List::create(
    Rcpp::_["beta_list"] = beta_out,
    Rcpp::_["Omega_list"] = Omega_out,
    Rcpp::_["Sigma_list"] = Sigma_out
  );
}

// -----------------------------------------------------------------------------
// Exact full ELBO for convergence monitoring.
// Because B_ig is already the full class-specific lower bound including
// constants, this function does not add any extra constants.
// -----------------------------------------------------------------------------
// [[Rcpp::export]]
double exact_elbo_gmix_flexible_cpp(arma::mat tau,
                                    arma::mat pi,
                                    arma::mat B){
  int n = tau.n_rows;
  int G = tau.n_cols;
  double val = 0.0;
  double eps = 1e-300;

  for(int i = 0; i < n; ++i){
    for(int g = 0; g < G; ++g){
      double t = tau(i, g);
      if(t <= 0.0) continue;
      val += t * (std::log(std::max(pi(i, g), eps)) + B(i, g) - std::log(std::max(t, eps)));
    }
  }
  return val;
}

// -----------------------------------------------------------------------------
// Extract conditional variational parameters from joint lists.
// This is mainly for diagnostics and for returning results to R.
// -----------------------------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List extract_variational_gmix_flexible_cpp(Rcpp::List Y_list,
                                                 Rcpp::List m_joint_list,
                                                 Rcpp::List C_joint_list,
                                                 int p,
                                                 int r){
  int n = Y_list.size();
  int G = m_joint_list.size();
  int db = p * r;

  Rcpp::List mu_b_out(G), S_b_out(G), A_out(G), L_out(G), V_out(G), m_eta_out(G), C_etaeta_out(G);

  for(int g = 0; g < G; ++g){
    Rcpp::List m_g = m_joint_list[g];
    Rcpp::List C_g = C_joint_list[g];
    Rcpp::List mu_g(n), S_g(n), A_g(n), L_g(n), V_g(n), meta_g(n), Cee_g(n);

    for(int i = 0; i < n; ++i){
      arma::mat Y = Rcpp::as<arma::mat>(Y_list[i]);
      int dEta = p * Y.n_cols;
      arma::vec m = Rcpp::as<arma::vec>(m_g[i]);
      arma::mat C = Rcpp::as<arma::mat>(C_g[i]);
      Rcpp::List conv = convert_joint_to_conditional(m, C, db, dEta);
      mu_g[i] = conv["mu_b"];
      S_g[i] = conv["S_b"];
      A_g[i] = conv["A"];
      L_g[i] = conv["L"];
      V_g[i] = conv["V"];
      meta_g[i] = conv["m_eta"];
      Cee_g[i] = conv["C_etaeta"];
    }

    mu_b_out[g] = mu_g;
    S_b_out[g] = S_g;
    A_out[g] = A_g;
    L_out[g] = L_g;
    V_out[g] = V_g;
    m_eta_out[g] = meta_g;
    C_etaeta_out[g] = Cee_g;
  }

  return Rcpp::List::create(
    Rcpp::_["mu_b_list"] = mu_b_out,
    Rcpp::_["S_b_list"] = S_b_out,
    Rcpp::_["A_list"] = A_out,
    Rcpp::_["L_list"] = L_out,
    Rcpp::_["V_list"] = V_out,
    Rcpp::_["m_eta_list"] = m_eta_out,
    Rcpp::_["C_etaeta_list"] = C_etaeta_out
  );
}
