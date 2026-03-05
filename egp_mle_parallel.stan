// =========================================================================
// OPTIMIZED STAN MODEL: Extended Generalized Pareto (EGP)
// =========================================================================

functions {
  // 1. Generalized Pareto (GP) Log-PDF 
  // Standard GPD log-density for exceedances
  real gp_lpdf(real m, real sigma, real xi) {
    if (sigma <= 0 || m < 0) return negative_infinity();
    
    if (abs(xi) > 1e-12) {
      real t = 1.0 + xi * m / sigma;
      if (t <= 0) return negative_infinity(); // Support constraint
      return -log(sigma) + (-1.0/xi - 1.0) * log(t);
    } else {
      return -log(sigma) - m / sigma; // Exponential case as xi -> 0
    }
  }

  //  2. GP Cumulative Distribution Function (CDF)
  // Used as the probability integral transform for EGP extensions
  real gp_cdf(real m, real sigma, real xi) {
    if (sigma <= 0 || m <= 0) return 0.0;
    
    if (abs(xi) > 1e-12) {
      real t = 1.0 + xi * m / sigma;
      if (t <= 0) return 1.0;
      return 1.0 - pow(t, -1.0/xi);
    } else {
      return 1.0 - exp(-m / sigma);
    }
  }

  // ---- 3. EGP Transformation Log-PDFs ----
  // Each ID corresponds to a different parametric form of the EGP

  // ID 0: Standard GP (No transformation)
  real egp0_lpdf(real log_d_gp, real p_gp, real kappa) {
    return log_d_gp;
  }

  // ID 1: EGP-Power (Naveau et al., 2016)
  real egp1_lpdf(real log_d_gp, real p_gp, real kappa) {
    if (p_gp <= 0) return negative_infinity();
    return log(kappa) + (kappa - 1.0) * log(p_gp) + log_d_gp;
  }

  // ID 2: EGP-Normal
  real egp2_lpdf(real log_d_gp, real p_gp, real kappa) {
    if (kappa <= 0) return negative_infinity();
    // Normalization constant for truncated normal CDF transform
    real log_term1 = log(2.0) + 0.5 * log(kappa) - log(2.0 * normal_cdf(sqrt(kappa) | 0, 1) - 1.0);
    real log_term2 = normal_lpdf(sqrt(kappa) * (p_gp - 1.0) | 0, 1);
    return log_term1 + log_term2 + log_d_gp;
  }

  // ID 3: EGP-Beta (Gamet & Jalbert, 2022)
  real egp3_lpdf(real log_d_gp, real p_gp, real k1, real k2, real lb) {
    real ub = (k1 * k2 - 1.0) / (k2 - 2.0); // Derived upper bound
    real inner_val = p_gp * (ub - lb) + lb;
    
    real log_term1 = beta_lpdf(inner_val | k1, k2) 
                     + log(ub - lb) 
                     - log(beta_cdf(ub | k1, k2) - beta_cdf(lb | k1, k2));
    return log_term1 + log_d_gp;
  }

  // ID 4: EGP-Generalized Beta
  real gbeta1k_lpdf(real u, real a, real b, real c) {
    if (u <= 0 || u >= 1) return negative_infinity();
    return log(c) - lbeta(a, b) + (a * c - 1.0) * log(u) + (b - 1.0) * log1m(pow(u, c));
  }

  real gbeta1k_cdf(real u, real a, real b, real c) {
    return beta_cdf(pow(u, c) | a, b);
  }

  real egp4_lpdf(real log_d_gp, real p_gp, real k1, real k2, real lb, real k3) {
    real ub = pow((k1 * k3 - 1.0) / (k1 * k3 + k2 * k3 - k3 - 1.0), 1.0 / k3);
    real inner_val = p_gp * (ub - lb) + lb;
    
    real log_term1 = gbeta1k_lpdf(inner_val | k1, k2, k3) 
                 + log(ub - lb) 
                 - log(gbeta1k_cdf(ub | k1, k2, k3) - gbeta1k_cdf(lb | k1, k2, k3));
    return log_term1 + log_d_gp;
  }

  // 4. Partial Log-Likelihood for reduce_sum 
  // Function to compute log-density in parallel slices
  real partial_egp_lpdf(array[] real m_slice, int start, int end, 
                        array[] real s_slice, int egp_id, int season_form,
                        real log_sigma0, real xi_0,
                        real season_a, real season_t, real season_d,
                        real k1, real k2, real lb, real k3) {
    real pt_sum = 0.0;
    
    for (i in 1:size(m_slice)) {
      real m = m_slice[i];
      real s = s_slice[i];
      
      // Seasonal covariate g(s) calculation (Skewed Sine Wave)
      real g;
      if (abs(season_t) < 1e-12) {
        g = sin(s - season_d);
      } else {
        g = (1.0 / season_t) * tanh(season_t * sin(s - season_d) 
                                   / (1.0 - season_t * cos(s - season_d)));
      }

      // Parameter POSITIVITY: Scale (sigma) must be positive, enforced by exp()
      real sigma = (season_form == 1) ? exp(log_sigma0 + season_a * g) : exp(log_sigma0);
      
      // Shape (xi) can have additive seasonal effects
      real xi = (season_form == 2) ? (xi_0 + season_a * g) : xi_0;

      // Core GP components
      real log_d_gp = gp_lpdf(m | sigma, xi);
      real p_gp     = gp_cdf(m | sigma, xi);

      if (is_inf(log_d_gp)) {
         pt_sum += negative_infinity(); 
         continue; 
      }

      // Distribute likelihood based on selected EGP model ID
      if (egp_id == 0)      pt_sum += egp0_lpdf(log_d_gp | p_gp, k1);
      else if (egp_id == 1) pt_sum += egp1_lpdf(log_d_gp | p_gp, k1);
      else if (egp_id == 2) pt_sum += egp2_lpdf(log_d_gp | p_gp, k1);
      else if (egp_id == 3) pt_sum += egp3_lpdf(log_d_gp | p_gp, k1, k2, lb);
      else if (egp_id == 4) pt_sum += egp4_lpdf(log_d_gp | p_gp, k1, k2, lb, k3);
      else if (egp_id == 5) pt_sum += egp3_lpdf(log_d_gp | p_gp, k1, k1, 1.0/32.0);
    }
    return pt_sum;
  }
}

data {
  int<lower=0> N;
  vector[N] y;           // Observed rainfall time series
  vector[N] s;           // Seasonal phase mapping [0, 2*pi]

  real u;                // Exceedance threshold
  int egp_id;            // Model selector (0-5)
  int season_form;       // Seasonality mode (0:None, 1:Sigma, 2:Xi)
}

transformed data {
  // Count and extract only values exceeding threshold (m = y - u)
  int<lower=0> N_exc = 0;
  for (i in 1:N) {
    if (y[i] > u) N_exc += 1;
  }

  array[N_exc] real m_exc;
  array[N_exc] real s_exc;
  int pos = 1;
  for (i in 1:N) {
    if (y[i] > u) {
      m_exc[pos] = y[i] - u;
      s_exc[pos] = s[i];
      pos += 1;
    }
  }
  
  real eps = 1e-9;
  int grainsize = max(1, N_exc / 4); // Slicing size for parallel computation
}

parameters {
  real log_sigma0;        // Base scale on log-scale
  real xi_0;              // Base shape parameter

  // Seasonal parameters
  real<lower=0> season_a;               // Amplitude
  real<lower=-1, upper=1> season_t;      // Skewness
  real<lower=0> season_d;               // Phase

  // EGP shape parameters constrained to [0,1]
  real<lower=eps, upper=1> kappa1;
  real<lower=eps, upper=1> kappa2;
  real<lower=eps, upper=0.1> kappa3;
  real<lower=eps, upper=1> kappa4;
}

model {
  //  Priors 
  log_sigma0 ~ normal(0, 1);
  xi_0 ~ normal(0, 0.2);

  if (season_form != 0) {
    season_a ~ gamma(2, 1);
    season_t ~ normal(0, 1); 
    season_d ~ gamma(2, 1);
  }

  // Informatie priors for EGP shape parameters
  kappa1 ~ beta(2, 4); 
  kappa2 ~ beta(2, 4);
  kappa4 ~ beta(2, 4);
  kappa3 ~ gamma(2, 40); 

  // Likelihood (Multi-threaded reduce_sum) 
  target += reduce_sum(
    partial_egp_lpdf, m_exc, grainsize, 
    s_exc, egp_id, season_form,         
    log_sigma0, xi_0, season_a, season_t, season_d, 
    kappa1, kappa2, kappa3, kappa4
  );
}

generated quantities {
  // Pointwise Log-Likelihood for LOO-CV cross-validation
  vector[N_exc] log_lik;

  for (i in 1:N_exc) {
    log_lik[i] = partial_egp_lpdf(
      {m_exc[i]} | 1, 1, 
      {s_exc[i]}, egp_id, season_form,
      log_sigma0, xi_0, season_a, season_t, season_d,
      kappa1, kappa2, kappa3, kappa4
    );
  }
}

