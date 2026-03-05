# ==============================================================================
# SCRIPT 01: Threshold Stability Check
# Goal: Fit a standard GP model (EGP ID = 0) across different thresholds (u)
#       to identify the optimal threshold where the shape parameter (xi) stabilizes.
# ==============================================================================

library(tidyverse)
library(cmdstanr)
library(posterior)
library(ggplot2)

# --- 1. Data Preparation ------------------------------------------------------
cat("Loading and preparing precipitation data...\n")

my_data <- read_csv("historical_precipitation_fixed.csv", show_col_types = FALSE) %>%
  select(t = 1, y = 2) %>% 
  mutate(t = as.Date(t)) %>%
  drop_na(y) %>%
  group_by(t) %>%
  summarise(y = sum(y, na.rm = TRUE), .groups = "drop") %>%
  # Create a seasonal covariate (s) mapped to [0, 2*pi]
  mutate(s = yday(t) / (365 + leap_year(year(t))) * 2 * pi) 

cat("Compiling Stan model...\n")
model_egp <- cmdstan_model("egp_mle_parallel.stan", cpp_options = list(stan_threads = TRUE))

# --- 2. Threshold Sensitivity Analysis ----------------------------------------
thresholds <- seq(10, 20, by = 2) 
my_seed <- 12345
results_list <- list()

cat("Starting MCMC fits for threshold sensitivity...\n")

for (u_val in thresholds) {
  n_exceed <- sum(my_data$y > u_val)
  cat(sprintf("   - Fitting Threshold u = %d (n = %d)...\n", u_val, n_exceed))
  
  stan_data <- list(
    N = nrow(my_data), 
    y = my_data$y, 
    s = my_data$s,
    u = u_val,       
    egp_id = 0,      # Standard GP (Baseline)
    season_form = 0  # No seasonality for stability check
  )
  
  fit_test <- model_egp$sample(
    data = stan_data, 
    seed = my_seed,             
    chains = 4, 
    parallel_chains = 4, 
    threads_per_chain = 2,
    iter_warmup = 1000, 
    iter_sampling = 1000, 
    adapt_delta = 0.95, 
    refresh = 0,
    show_messages = FALSE
  )
  
  summary_stats <- fit_test$summary(variables = c("xi_0", "log_sigma0"))
  
  results_list[[as.character(u_val)]] <- tibble(
    threshold = u_val,
    n_exceed  = n_exceed,
    xi_mean   = summary_stats$mean[summary_stats$variable == "xi_0"],
    xi_q5     = summary_stats$q5[summary_stats$variable == "xi_0"],
    xi_q95    = summary_stats$q95[summary_stats$variable == "xi_0"],
    sig_mean  = summary_stats$mean[summary_stats$variable == "log_sigma0"],
    sig_q5    = summary_stats$q5[summary_stats$variable == "log_sigma0"],
    sig_q95   = summary_stats$q95[summary_stats$variable == "log_sigma0"]
  )
}

stability_data <- bind_rows(results_list)

# --- 3. Parameter Stability Plot ----------------------------------------------
cat("\nGenerating parameter stability plot...\n")

plot_data <- stability_data %>%
  pivot_longer(
    cols = c(starts_with("xi"), starts_with("sig")),
    names_to = c("param", ".value"),
    names_pattern = "(xi|sig)_(.*)"
  ) %>%
  mutate(param_label = ifelse(param == "xi", "1. Shape Parameter (xi)", "2. Scale Parameter (log_sigma)"))

p_stability <- ggplot(plot_data, aes(x = threshold, y = mean)) +
  geom_point(size = 3, color = "dodgerblue") + 
  geom_line(color = "dodgerblue", linewidth = 1) +
  geom_errorbar(aes(ymin = q5, ymax = q95), width = 0.4, color = "gray50") +
  geom_text(data = filter(plot_data, param == "xi"),
            aes(label = paste0("n=", n_exceed), y = q95 + abs(q95)*0.1), 
            size = 3.5, color = "red") +
  facet_wrap(~ param_label, scales = "free_y", ncol = 1) +
  labs(
    title = "Threshold Stability Comparison (Baseline GP Model)",
    x = "Threshold (u)",
    y = "Posterior Mean (with 90% CI)"
  ) +
  scale_x_continuous(breaks = thresholds) +
  theme_minimal(base_size = 14) +
  theme(
    strip.text = element_text(face = "bold", size = 12, hjust = 0),
    panel.border = element_rect(color = "gray80", fill = NA),
    plot.title = element_text(face = "bold")
  )

ggsave("01_Threshold_Stability_Plot.png", p_stability, width = 8, height = 8, bg = "white")
write_csv(stability_data, "01_Threshold_Diagnostics.csv")

cat("Script 01 finished. Results saved to working directory.\n")