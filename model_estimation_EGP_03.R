# ==============================================================================
# SCRIPT 03: Final Model Estimation and MCMC Diagnostics
# Goal: Run long MCMC chains for the winning model (Standard GP + Seasonality),
#       perform rigorous diagnostic checks, and export parameters for the 
#       downstream Hawkes-Copula pipeline.
# ==============================================================================

library(tidyverse)
library(lubridate)
library(cmdstanr)
library(posterior)
library(bayesplot)
library(ggplot2)

# Set bayesplot 
color_scheme_set("mix-blue-red")
theme_set(theme_minimal(base_size = 14))

# 1. Data Preparation

my_data <- read_csv("historical_precipitation_fixed.csv", show_col_types = FALSE) %>%
  select(t = 1, y = 2) %>%
  mutate(t = as.Date(t)) %>%
  drop_na(y) %>%
  group_by(t) %>%
  summarise(y = sum(y, na.rm = TRUE), .groups = "drop") %>%
  # Seasonal covariate s mapped to [0, 2*pi]
  mutate(s = yday(t) / (365 + leap_year(year(t))) * 2 * pi)

# 2. Compile Model & Set Configurations 

model_egp <- cmdstan_model("egp_mle_parallel.stan", cpp_options = list(stan_threads = TRUE))

# Winning model parameters defined from prior scripts
best_u <- 12
best_egp_id <- 0       # 0: Standard GP
best_season_form <- 1  # 1: Seasonality enabled

stan_data_final <- list(
  N = nrow(my_data),
  y = my_data$y,
  s = my_data$s,
  u = best_u,
  egp_id = best_egp_id,
  season_form = best_season_form
)

# 3. Run Production-Ready MCMC 
cat(sprintf("Running final MCMC chains at threshold u = %d...\n", best_u))
fit_final <- model_egp$sample(
  data = stan_data_final,
  seed = 888,              
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = 2,
  iter_warmup = 2000,      
  iter_sampling = 4000,    
  adapt_delta = 0.99,      
  max_treedepth = 15,
  refresh = 500
)

# 4. Extract Diagnostics & Summaries

# Filter out hidden Stan variables, unused kappas, and pointwise log-likelihoods
fit_summary <- fit_final$summary() %>%
  filter(!str_detect(variable, "^lp__$|^lprior$")) %>%
  filter(!str_detect(variable, "^kappa")) %>%
  filter(!str_detect(variable, "^log_lik")) 

# Print clean summary table to console
print(fit_summary)

# Check for convergence warnings (divergences, treedepth, R-hat)
fit_final$cmdstan_diagnose()

# Export summary statistics to CSV
write_csv(fit_summary, paste0("03_Final_Model_Summary_u", best_u, ".csv"))

#  5. Visual Diagnostics (Trace & Density Plots)
cat("\nGenerating diagnostic plots...\n")

# Extract posterior draws for plotting
draws_array <- fit_final$draws()
params_to_plot <- fit_summary$variable

# 1. Trace plots: Check mixing and stationarity of chains
p_trace <- mcmc_trace(draws_array, pars = params_to_plot) +
  ggtitle(sprintf("MCMC Trace Plots (u = %d, GP + Seasonality)", best_u))
print(p_trace)
ggsave("03_Diagnostic_Trace.png", p_trace, width = 10, height = 8, bg = "white")

# 2. Posterior Density plots: Visualize parameter uncertainty
p_areas <- mcmc_areas(draws_array, pars = params_to_plot, prob = 0.8, prob_outer = 0.95) +
  ggtitle("Posterior Distributions with 80% & 95% Intervals")
print(p_areas)
ggsave("03_Diagnostic_Posteriors.png", p_areas, width = 8, height = 6, bg = "white")

# 3. Convergence Plots: R-hat and ESS (Effective Sample Size)
p_rhat <- mcmc_rhat(fit_summary$rhat) + yaxis_text() + ggtitle("R-hat Diagnostic")
ggsave("03_Diagnostic_Rhat.png", p_rhat, width = 8, height = 6, bg = "white")

p_ess <- mcmc_neff(fit_summary$ess_bulk / (4 * 4000)) + yaxis_text() + ggtitle("Effective Sample Size Ratio")
ggsave("03_Diagnostic_ESS.png", p_ess, width = 8, height = 6, bg = "white")

cat("MCMC diagnostics completed. Plots saved to working directory.\n")

# 6. Export Parameters for Hawkes and Copula Pipelines
cat("\nExporting 15-parameter 'egp_estim' vector for downstream pipeline...\n")

# Helper: Extract posterior mean for a specific parameter
get_mean <- function(param_name) {
  val <- fit_summary$mean[fit_summary$variable == param_name]
  if(length(val) == 0) return(0) else return(val)
}

# Construct a 15-element vector as required by the downstream Stan model
egp_estim_vector <- numeric(15)
egp_estim_vector[1]  <- get_mean("log_sigma0")
egp_estim_vector[2]  <- get_mean("xi_0")
egp_estim_vector[3]  <- get_mean("season_a")
egp_estim_vector[4]  <- get_mean("season_t")
egp_estim_vector[5]  <- get_mean("season_d")
egp_estim_vector[6]  <- 0 # kappa1: Not used in baseline GP
egp_estim_vector[7]  <- 0 # kappa2
egp_estim_vector[8]  <- 0 # kappa3
egp_estim_vector[9]  <- 0 # kappa4
egp_estim_vector[10] <- best_egp_id      
egp_estim_vector[11] <- best_season_form 
egp_estim_vector[12:15] <- 0 # Padded placeholders


# EXPORT 1: Parameter Summary Table (CSV with labels)

egp_params_table <- tibble(
  index = 1:15,
  parameter = c("log_sigma0", "xi_0", "season_a", "season_t", "season_d", 
                "kappa1", "kappa2", "kappa3", "kappa4", 
                "egp_id", "season_form", 
                "pad12", "pad13", "pad14", "pad15"),
  value = egp_estim_vector
)

summary_file <- paste0("03_marginal_params_summary_u", best_u, ".csv")
write_csv(egp_params_table, summary_file)
cat("Parameter summary table exported to:", summary_file, "\n")


# EXPORT 2: Estimation Vector for Stan Input 

# Format A: Plain CSV array (Single row, no headers)
vector_csv_file <- paste0("03_egp_estim_vector_u", best_u, ".csv")
write.table(t(egp_estim_vector), file = vector_csv_file, 
            row.names = FALSE, col.names = FALSE, sep = ",")
cat("Estimation vector (CSV) exported to:", vector_csv_file, "\n")

# Format B: RDS object (Standard R format)
vector_rds_file <- paste0("03_egp_estim_vector_u", best_u, ".rds")
saveRDS(egp_estim_vector, vector_rds_file)
cat("Estimation vector (RDS) exported to:", vector_rds_file, "\n")

cat("\nScript 03 execution finalized successfully.\n")