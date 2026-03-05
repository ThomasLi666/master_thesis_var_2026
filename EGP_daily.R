
# PART 1: SETUP & DATA PREPARATION

library(tidyverse)
library(cmdstanr)
library(loo)      
library(posterior)
library(bayesplot) 
library(ggplot2)

# Global setting for bayesplot traceplot color scheme
color_scheme_set("mix-blue-red")

# Read and clean historical precipitation data
my_data <- read_csv("historical_precipitation_fixed.csv", show_col_types = FALSE) %>%
  select(t = 1, y = 2) %>% 
  mutate(t = as.Date(t)) %>%
  drop_na(y) %>%
  group_by(t) %>%
  summarise(y = sum(y, na.rm = TRUE), .groups = "drop") %>%
  mutate(
    # Create seasonal covariate s mapped to [0, 2*pi] 
    s = yday(t) / (365 + leap_year(year(t))) * 2 * pi 
  )

# Compile the Stan model for EGP inference
model_egp <- cmdstan_model("egp_mle_parallel.stan", cpp_options = list(stan_threads = TRUE))


# PART 2: THRESHOLD SENSITIVITY CHECK (Explore u from 10 to 20)


# Define threshold range for sensitivity analysis 
thresholds <- seq(10, 20, by = 2) 
xi_results <- tibble() 

# Set a global seed for reproducibility 
my_seed <- 12345 

for (u_val in thresholds) {
  
  # Calculate number of exceedances above current threshold u 
  n_exceed <- sum(my_data$y > u_val)
  
  # Prepare data list for Stan model
  stan_data <- list(
    N = nrow(my_data), 
    y = my_data$y, 
    s = my_data$s,
    u = u_val,       # Current dynamic threshold
    egp_id = 2,      # EGP-Normal model ID 
    season_form = 1  # 1: Enable seasonal scale parameter sigma(t) 
  )
  
  # Run MCMC sampling with set seed for reproducible results
  fit_test <- model_egp$sample(
    data = stan_data, 
    seed = my_seed,              
    chains = 4, 
    parallel_chains = 4, 
    threads_per_chain = 2,
    iter_warmup = 800, 
    iter_sampling = 800, 
    adapt_delta = 0.95, 
    refresh = 0
  )
  
  # Extract detailed diagnostics for core parameters:
  # xi_0 (Shape), log_sigma0 (Scale), and kappa1 (EGP Transform)
  param_summary <- fit_test$summary(
    variables = c("xi_0", "log_sigma0", "kappa1")
  )
  
  # Initialize tibble row for current threshold u
  row_data <- tibble(
    threshold = u_val,
    n_exceed = n_exceed
  )
  
  # Automate merging of parameter statistics into a flat row
  # Includes Mean, Rhat, and ESS 
  for(p in c("xi_0", "log_sigma0", "kappa1")) {
    p_stats <- param_summary %>% filter(variable == p)
    
    row_data[[paste0(p, "_mean")]] <- p_stats$mean
    row_data[[paste0(p, "_rhat")]] <- p_stats$rhat      # MCMC convergence diagnostic
    row_data[[paste0(p, "_ess")]]  <- p_stats$ess_bulk  # Bulk Effective Sample Size
    row_data[[paste0(p, "_q5")]]   <- p_stats$q5        # Lower 90% credible limit
    row_data[[paste0(p, "_q95")]]  <- p_stats$q95       # Upper 90% credible limit
  }
  
  # Consolidate results into master table
  xi_results <- bind_rows(xi_results, row_data)
  
  # Extract MCMC draws for traceplot visualization
  draws_mcmc <- fit_test$draws(variables = c("log_sigma0", "xi_0", "kappa1"))
  
  # Generate traceplot to verify "hairy caterpillar" mixing 
  p_trace <- mcmc_trace(draws_mcmc, facet_args = list(ncol = 1)) +
    labs(title = paste("MCMC Traceplot for u =", u_val))
  
  # Save individual traceplots for visual diagnostic verification
  ggsave(paste0("Traceplot_Threshold_", u_val, ".png"), plot = p_trace, width = 8, height = 6, bg = "white")
}


# PART 3: EXPORT & VISUALIZATION

# Export final comprehensive CSV containing all parameters and convergence metrics
write_csv(xi_results, "Threshold_Full_Diagnostics.csv")
cat("\n✅ Full diagnostics table saved to 'Threshold_Full_Diagnostics.csv'\n")
print(xi_results) 

# Generate the Threshold Stability Plot for the Shape Parameter (xi)
p_stability <- ggplot(xi_results, aes(x = threshold, y = xi_0_mean)) +
  geom_point(size = 3, color = "dodgerblue") + 
  geom_line(color = "dodgerblue", linewidth = 1) +
  geom_errorbar(aes(ymin = xi_0_q5, ymax = xi_0_q95), width = 0.4, color = "gray50") +
  # Label exceedance counts (n) to visualize sample size trade-offs 
  geom_text(aes(label = paste0("n=", n_exceed), y = xi_0_q95 + 0.02), size = 3.5, color = "red") +
  labs(
    title = "Threshold Stability Plot (EGP_Normal)", 
    x = "Threshold (u)", 
    y = "Shape Parameter (xi_0) with 90% CI"
  ) +
  theme_minimal(base_size = 14)

# Save final parameter stability visualization
ggsave("Threshold_Stability_Plot.png", p_stability, width = 8, height = 6, bg = "white")


# PART 4: MULTI-PARAMETER THRESHOLD STABILITY COMPARISON PLOT

cat("\n=== PART 4: GENERATING MULTI-PARAMETER STABILITY PLOT ===\n")

# 1. Reshape the data for multi-panel plotting
plot_data <- xi_results %>%
  select(threshold, n_exceed, 
         xi_0_mean, xi_0_q5, xi_0_q95, 
         log_sigma0_mean, log_sigma0_q5, log_sigma0_q95, 
         kappa1_mean, kappa1_q5, kappa1_q95) %>%
  pivot_longer(
    cols = -c(threshold, n_exceed),
    names_to = c("parameter", ".value"),
    names_pattern = "(.*)_(mean|q5|q95)" # Regex to split parameter name and stat type
  ) %>%
  # Rename parameters for cleaner facet labels in the plot
  mutate(parameter = case_when(
    parameter == "xi_0" ~ "1. Shape Parameter (xi_0)",
    parameter == "log_sigma0" ~ "2. Scale Parameter (log_sigma0)",
    parameter == "kappa1" ~ "3. EGP Transform (kappa1)",
    TRUE ~ parameter
  ))

# 2. Create the multi-panel comparison plot
p_comp_stability <- ggplot(plot_data, aes(x = threshold, y = mean)) +
  geom_point(size = 3, color = "dodgerblue") + 
  geom_line(color = "dodgerblue", linewidth = 1) +
  geom_errorbar(aes(ymin = q5, ymax = q95), width = 0.4, color = "gray50") +
  
  # Add exceedance count (n) ONLY to the top facet (Shape parameter) to avoid clutter
  geom_text(data = filter(plot_data, parameter == "1. Shape Parameter (xi_0)"),
            aes(label = paste0("n=", n_exceed), y = q95 + abs(q95)*0.1), 
            size = 3.5, color = "red") +
  
  # Create a separate panel for each parameter, stacked vertically
  facet_wrap(~ parameter, scales = "free_y", ncol = 1) +
  
  # Labels and themes
  labs(
    title = "Comprehensive Threshold Stability Comparison (EGP_Normal)",
    x = "Threshold (u)",
    y = "Posterior Mean with 90% CI"
  ) +
  scale_x_continuous(breaks = thresholds) +
  theme_minimal(base_size = 14) +
  theme(
    strip.text = element_text(face = "bold", size = 12, hjust = 0), # Bold facet titles
    panel.border = element_rect(color = "gray80", fill = NA),       # Add borders around facets
    plot.title = element_text(face = "bold")
  )

# 3. Save and display the plot
ggsave("Threshold_Stability_Comparison_Plot.png", p_comp_stability, width = 10, height = 10, bg = "white")
print(p_comp_stability)

cat("✅ Multi-parameter stability plot saved to 'Threshold_Stability_Comparison_Plot.png'\n")




# PART 5: FINAL MODEL FITTING & PARAMETER EXTRACTION (u = 12)


# 1. Define the final chosen configuration based on your stability checks
final_u <- 12
final_egp_id <- 2       # EGP normal
final_season_form <- 1  # Seasonality on scale (sigma)

# 2. Prepare data for the final run
stan_data_final <- list(
  N = nrow(my_data), 
  y = my_data$y, 
  s = my_data$s,
  u = final_u, 
  egp_id = final_egp_id, 
  season_form = final_season_form,
  xi_season_form = 0  
)

# 3. Run MCMC with high precision
fit_final <- model_egp$sample(
  data = stan_data_final, 
  seed = 2026, 
  chains = 4, 
  parallel_chains = 4, 
  threads_per_chain = 2,
  iter_warmup = 1500,    
  iter_sampling = 1500,  
  adapt_delta = 0.99, 
  refresh = 500
)

# 4. Extract posterior means for all parameters
param_summary <- fit_final$summary()
post_means <- param_summary$mean
names(post_means) <- param_summary$variable

# Helper function to safely extract parameters (returns 0 if not used in the model)
get_param <- function(p_name) {
  if (p_name %in% names(post_means)) return(post_means[p_name])
  return(0.0)
}

# 5. Construct the strictly formatted 15-dimensional vector
egp_estim_15 <- rep(0, 15)

# Fill core parameters (Note: log_sigma0 MUST be exponentiated back to the linear scale)
egp_estim_15[1] <- exp(get_param("log_sigma0"))  # sigma_0
egp_estim_15[2] <- get_param("xi_0")             # xi_0
egp_estim_15[3] <- get_param("season_a")         # season_a
egp_estim_15[4] <- get_param("season_t")         # season_t
egp_estim_15[5] <- get_param("season_d")         # season_d

# Fill EGP-specific transformation parameters
egp_estim_15[6] <- get_param("kappa1")           # kappa1 (Used by EGP-Normal)
egp_estim_15[7] <- get_param("kappa2")           # kappa2
egp_estim_15[8] <- get_param("kappa3")           # kappa3
egp_estim_15[9] <- get_param("kappa4")           # kappa4

# Fill model control flags
egp_estim_15[10] <- final_egp_id                 # egp_id
egp_estim_15[11] <- final_season_form            # season_form

# Add names 
names(egp_estim_15) <- c(
  "sigma_0", "xi_0", "season_a", "season_t", "season_d",
  "kappa1", "kappa2", "kappa3", "kappa4",
  "egp_id", "season_form", "pad1", "pad2", "pad3", "pad4"
)

# Print the vector to the console 
print(round(egp_estim_15, 4))

# 6. Save as an .rds file for the Hawkes process script
saveRDS(egp_estim_15, "egp_params_u12.rds")
cat("\n✅ SUCCESS: 15D parameter vector saved to 'egp_params_u12.rds'.\n")