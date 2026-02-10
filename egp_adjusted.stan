// egp.stan - Optimized for Gävle Daily Data 

functions {
  // Basic GP
  real gp_pdf(real m, real sigma, real xi) {
    real dens;
    if (xi != 0)
      dens = 1/sigma * (1 + xi*m/sigma)^(-1/xi-1);
    else
      dens = 1/sigma * exp(-m/sigma);
    return dens;
  }
  real gp_cdf(real m, real sigma, real xi) {
    real prob;
    if (xi != 0)
      prob = 1 - (1 + xi*m/sigma)^(-1/xi);
    else
      prob = 1 - exp(-m/sigma);
    return prob;
  }
  real gp_q(real p, real sigma, real xi) {
    real quant;
    if (xi != 0)
      quant = sigma * ((1 - p)^(-xi) - 1)/xi;
    else
      quant = sigma * log(1 - p);
    return quant;
  }

  //  EGP Extended Distribution Family
  // _e0: no EGP
  real egp0_pdf(real d_gp, real p_gp, real kappa) {
    return d_gp;
  }
  
  // _e1: EGP-power
  real egp1_pdf(real d_gp, real p_gp, real kappa) {
    return kappa * p_gp^(kappa-1) * d_gp;
  }

  // _e2: EGP-normal
  real egp2_pdf(real d_gp, real p_gp, real kappa) {
    real dens;
    if (kappa == 0)
      dens = 1;
    else {
      real term1 = 2*sqrt(kappa) / (2*normal_cdf(sqrt(kappa) | 0, 1) - 1);
      real term2 = exp(normal_lpdf(sqrt(kappa)*(p_gp - 1) | 0, 1));
      dens = term1 * term2 * d_gp;
    }
    return dens;
  }
  
  // _e3: EGP-beta
  real egp3_pdf(real d_gp, real p_gp, real k1, real k2, real lb) {
    real dens;
    real ub = (k1*k2-1) / (k2-2); 
    real term1 = exp(beta_lpdf(p_gp*(ub-lb) + lb | k1, k2)) * (ub-lb) / (beta_cdf(ub | k1, k2) - beta_cdf(lb | k1, k2));
    dens = term1 * d_gp;
    return dens;
  }
  
  // _e4: EGP-genbeta helpers
  real gbeta1k_pdf(real u, real a, real b, real c) {
    return c * beta(a, b)^(-1) * u^(a*c-1) * (1-u^c)^(b-1);
  }
  real gbeta1k_cdf(real u, real a, real b, real c) {
    return beta_cdf(u^c | a, b);
  }
  // _e4: EGP-genbeta
  real egp4_pdf(real d_gp, real p_gp, real k1, real k2, real lb, real k3) {
    real dens;
    real ub = ((k1*k3 - 1) / (k1*k3 + k2*k3 - k3 - 1))^(1/k3);
    real term1 = gbeta1k_pdf(p_gp*(ub-lb) + lb, k1, k2, k3) * (ub-lb) / (gbeta1k_cdf(ub | k1, k2, k3) - gbeta1k_cdf(lb | k1, k2, k3));
    dens = term1 * d_gp;
    return dens;
  }

  // Core Wrapper Functions 
  real egp_pdf(
    real m, real s,
    int egp_id, int season_form,
    real sigma_0, real xi_0,
    real season_a, real season_t, real season_d,
    real k1, real k2, real lb, real k3) {
    
    // Calculate seasonal Sigma
    real sigma;
    if (season_form == 1) { 
      if (season_t == 0)
        sigma = sigma_0 + season_a * sin(s - season_d);
      else
        sigma = sigma_0 + season_a * 1/season_t * tanh(season_t*sin(s - season_d) / (1 - season_t*cos(s - season_d)));
    }
    else
      sigma = sigma_0;
    
    // Calculate Seasonality Xi
    real xi;
    if (season_form == 2) {
      if (season_t == 0)
        xi = xi_0 + season_a * sin(s - season_d);
      else
        xi = xi_0 + season_a * 1/season_t * tanh(season_t*sin(s - season_d) / (1 - season_t*cos(s - season_d)));
    }
    else
      xi = xi_0;
      
    // 
    
    // Calculate GP
    real gp_dens = gp_pdf(m, sigma, xi);
    real gp_prob = gp_cdf(m | sigma, xi);
    
    // Apply EGP extension
    real dens;
    if (egp_id == 0) dens = egp0_pdf(gp_dens, gp_prob, k1);
    else if (egp_id == 1) dens = egp1_pdf(gp_dens, gp_prob, k1);
    else if (egp_id == 2) dens = egp2_pdf(gp_dens, gp_prob, k1);
    else if (egp_id == 3) dens = egp3_pdf(gp_dens, gp_prob, k1, k2, lb);
    else if (egp_id == 4) dens = egp4_pdf(gp_dens, gp_prob, k1, k2, lb, k3);
    else if (egp_id == 5) dens = egp3_pdf(gp_dens, gp_prob, k1, k1, 1.0/32.0);
    
    return dens;
  }
}

data {
  int N;
  vector[N] y; // time series
  vector[N] s; // season on [0, 2*pi]
  real u;      // threshold
  int egp_id;  // identifier of the distribution extending the GP
  int season_form; //0: no season effect, 1: sigma, 2: xi
}

transformed data {
  vector[N] m;        // mark
  array[N] int exc;   // counting exceedances
  
  m = y - u;          
  for (i in 1:N) {
    if (m[i] < 0) {
      m[i] = 0;
    }
    exc[i] = m[i] > 0;
  }
}

parameters {
  real sigma_0; 
  real xi_0; 
  
  real<lower=0> season_a; 
  real<lower=-1, upper=1> season_t; 
  real<lower=0> season_d; 
  
  real<lower=1e-9, upper=1> kappa1;
  real<lower=1e-9, upper=1> kappa2;
  real<lower=1e-9, upper=0.1> kappa3; 
  real<lower=1e-9, upper=1> kappa4;
}

model {
  // --- Priors ---
  if (season_form == 1) {
    sigma_0 ~ normal(0, 1);
    season_a ~ gamma(2, 1);
    season_t ~ normal(0, 1);
    season_d ~ gamma(2, 1);
  }
  else
    sigma_0 ~ gamma(2, 1);
  
  if (season_form == 2) {
    season_a ~ gamma(2, 1);
    season_t ~ normal(0, 1);
    season_d ~ gamma(2, 1);
  }
  
  // 放宽 Xi 先验
  xi_0 ~ normal(0, 0.5); 
  
  kappa1 ~ gamma(5, 15); 
  kappa2 ~ gamma(5, 15);
  kappa3 ~ gamma(2, 40); 
  kappa4 ~ gamma(5, 15);
  
  // --- Likelihood ---
  for (i in 1:N) {
    if (m[i] > 0) {
      target += log(egp_pdf(
        m[i], s[i],
        egp_id, season_form,
        sigma_0, xi_0,
        season_a, season_t, season_d,
        kappa1, kappa2, kappa3, kappa4
      ));
    }
  }
}

generated quantities {
  // 计算 BIC
  int egpid = egp_id;
  int seasonform = season_form;
  
  real loglik = 0;
  for (i in 1:N) {
    if (m[i] > 0) {
      loglik += log(egp_pdf(
        m[i], s[i],
        egp_id, season_form,
        sigma_0, xi_0,
        season_a, season_t, season_d,
        kappa1, kappa2, kappa3, kappa4
      ));
    }
  }

  int npar_egp;
  if (egp_id == 0) npar_egp = 0;
  else if (egp_id == 1) npar_egp = 1;
  else if (egp_id == 2) npar_egp = 1;
  else if (egp_id == 3) npar_egp = 3;
  else if (egp_id == 4) npar_egp = 4;
  else if (egp_id == 5) npar_egp = 1;
  
  if (season_form != 0)
   npar_egp = npar_egp + 3; 
  
  npar_egp = npar_egp + 2; 

  int n_exceedances = 0;
  for (i in 1:N) {
      if (exc[i] == 1) n_exceedances += 1;
  }
  
  real AIC = 2 * npar_egp - 2 * loglik;
  real BIC = npar_egp * log(n_exceedances) - 2 * loglik;
}

