# ==============================================================================
# SCRIPT 02: Model Selection
# Goal: Compare different Extended Generalized Pareto (EGP) models using LOO-CV
#       at the chosen threshold (u = 12), followed by a sensitivity check.
# ==============================================================================

library(tidyverse)
library(lubridate)
library(cmdstanr)
library(loo)

# 1. Data Preparation 

my_data <- read_csv("historical_precipitation_fixed.csv", show_col_types = FALSE) %>%
  select(t = 1, y = 2) %>%
  mutate(t = as.Date(t)) %>%
  drop_na(y) %>%
  group_by(t) %>%
  summarise(y = sum(y, na.rm = TRUE), .groups = "drop") %>%
  mutate(s = yday(t) / (365 + leap_year(year(t))) * 2 * pi)

model_egp <- cmdstan_model("egp_mle_parallel.stan", cpp_options = list(stan_threads = TRUE))

# 2. Define Models and Helper Function
egp_models <- c(
  GP_Standard = 0,
  EGP_Power   = 1,
  EGP_Normal  = 2,
  EGP_Beta    = 3,
  EGP_GenBeta = 4
)

fit_and_loo <- function(u, egp_id, season_form = 1, seed = 123,
                        chains = 4, warmup = 1000, sampling = 1000, adapt_delta = 0.95) {
  
  stan_data <- list(
    N = nrow(my_data),
    y = my_data$y,
    s = my_data$s,
    u = u,
    egp_id = egp_id,
    season_form = season_form
  )
  
  fit <- model_egp$sample(
    data = stan_data,
    seed = seed,
    chains = chains,
    parallel_chains = chains,
    threads_per_chain = 2,
    iter_warmup = warmup,
    iter_sampling = sampling,
    adapt_delta = adapt_delta,
    refresh = 0,              
    show_messages = FALSE     
  )
  
  loo_obj <- fit$loo(cores = min(4, chains))
  return(list(fit = fit, loo = loo_obj))
}

# 3. Main Model Selection (u = 12)
u_main <- 12
fits <- list()
loos <- list()

for (m in names(egp_models)) {
  cat(sprintf("   - Fitting model: %s (egp_id = %d)...\n", m, egp_models[[m]]))
  
  res <- fit_and_loo(u = u_main, egp_id = egp_models[[m]], season_form = 1)
  fits[[m]] <- res$fit
  loos[[m]] <- res$loo
}

# Compare models using LOO-CV
comp <- loo_compare(loos)
print(comp, simplify = FALSE)

# Export comparison results
comp_df <- as.data.frame(comp) %>% rownames_to_column("model")
write_csv(comp_df, paste0("02_model_comparison_u", u_main, ".csv"))
cat(sprintf("Model comparison saved to '02_model_comparison_u%d.csv'.\n", u_main))

best_model <- rownames(comp)[1]
cat(sprintf("Best model at u = %d based on LOO-IC: %s\n", u_main, best_model))

# 4. Sensitivity Check (u = 14) 
u_sens <- 14
top2 <- c(GP_Standard = 0, EGP_Normal = 2)

fits_s <- list()
loos_s <- list()


for (m in names(top2)) {
  cat(sprintf("   - Fitting model: %s (egp_id = %d)...\n", m, top2[[m]]))
  res <- fit_and_loo(u = u_sens, egp_id = top2[[m]], season_form = 1, seed = 456)
  fits_s[[m]] <- res$fit
  loos_s[[m]] <- res$loo
}


comp_s <- loo_compare(loos_s)
print(comp_s, simplify = FALSE)

comp_s_df <- as.data.frame(comp_s) %>% rownames_to_column("model")
write_csv(comp_s_df, paste0("02_model_comparison_u", u_sens, "_top2.csv"))
cat(sprintf("Sensitivity comparison saved to '02_model_comparison_u%d_top2.csv'.\n", u_sens))

