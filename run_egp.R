library(tidyverse)
library(lubridate)
library(cmdstanr)
library(bayesplot)
library(ggplot2)
library(posterior)

# 1. Data Import & Preprocessing


# Read data
df <- read_csv("historical_precipitation_fixed.csv", show_col_types = FALSE)

# Preprocessing: Calculate seasonality 's'
df_stan <- df %>%
  mutate(
    Date = ymd(Date), 
    day_idx = yday(Date),
    days_in_year = if_else(leap_year(Date), 366, 365),
    # Calculate radians s [0, 2pi]
    s = (day_idx / days_in_year) * 2 * pi
  ) %>%
  arrange(Date)

# 2. Analysis Function

run_egp_analysis <- function(id, data, model_file, df_raw) {
  
  model_names <- c("0_GP", "1_Power", "2_Normal", "3_Beta", "4_GenBeta")
  curr_name <- model_names[id + 1]
  
  cat(paste0("\nRunning model: ", curr_name, "...\n"))
  
  # Set EGP ID
  data$egp_id <- id
  
  # Run Sampling 
  fit <- model_file$sample(
    data = data,
    chains = 4,
    parallel_chains = 4,
    iter_warmup = 500,
    iter_sampling = 1000,
    refresh = 200,      # Show progress every 200 iters
    show_messages = FALSE
  )
  
  # Extraction & Diagnostics 
  
  # 1. AIC/BIC
  metrics <- fit$draws(variables = c("AIC", "BIC"), format = "df")
  mean_bic <- mean(metrics$BIC)
  
  # 2. Rhat & ESS check
  check_vars <- c("sigma_0", "xi_0", "season_a")
  if(id > 0) check_vars <- c(check_vars, "kappa1") 
  
  summ <- fit$summary(variables = check_vars)
  max_rhat <- max(summ$rhat, na.rm = TRUE)
  min_ess  <- min(summ$ess_bulk, na.rm = TRUE)
  
  # 3. Parameters
  post_mean <- fit$summary()
  sigma_est <- post_mean$mean[post_mean$variable == "sigma_0"]
  xi_est    <- post_mean$mean[post_mean$variable == "xi_0"]
  
  cat(paste0("   Finished. Rhat: ", round(max_rhat, 3), " | BIC: ", round(mean_bic, 1), "\n"))
  
  # Plotting 
  
  # Trace Plot
  p_trace <- mcmc_trace(fit$draws(check_vars)) +
    ggtitle(paste("Trace:", curr_name, "| Max Rhat:", round(max_rhat, 3)))
  ggsave(paste0("plot_trace_", curr_name, ".png"), p_trace, width = 8, height = 5)
  
  # QQ Plot
  threshold <- data$u
  exceedances <- df_raw$Precipitation[df_raw$Precipitation > threshold] - threshold
  n <- length(exceedances)
  p_idx <- (1:n) / (n + 1)
  theoretical_q <- (sigma_est / xi_est) * ((1 - p_idx)^(-xi_est) - 1)
  
  qq_data <- data.frame(Theo = theoretical_q, Emp = sort(exceedances))
  p_qq <- ggplot(qq_data, aes(x = Theo, y = Emp)) +
    geom_point(alpha = 0.5, color = "blue") +
    geom_abline(color = "red", linetype = "dashed") +
    labs(title = paste("QQ Plot:", curr_name)) + theme_minimal()
  ggsave(paste0("plot_qq_", curr_name, ".png"), p_qq, width = 5, height = 5)
  
  # Return Results 
  return(data.frame(
    Model_ID = id,
    Model_Name = curr_name,
    Max_Rhat = max_rhat,
    Min_ESS = min_ess,
    BIC = mean_bic,
    Sigma = sigma_est,
    Xi = xi_est
  ))
}

# 3. Main Execution

# Compile Stan model
mod <- cmdstan_model("egp_adjusted.stan")

# Base data list
base_data <- list(
  N = nrow(df_stan),
  y = df_stan$Precipitation,
  s = df_stan$s,
  u = 1.0,            
  season_form = 1
)

# Run batch 
results_list <- list()
for (id in 0:4) {
  results_list[[id + 1]] <- run_egp_analysis(id, base_data, mod, df_stan)
}

# Combine results
final_results <- bind_rows(results_list) %>%
  arrange(BIC)

# Print Summary
print("--- Model Comparison Table ---")
print(final_results)

# Plot Comparison
p_comp <- ggplot(final_results, aes(x = reorder(Model_Name, BIC), y = BIC, fill = Model_Name)) + 
  geom_col() +
  geom_text(aes(label = round(BIC, 1)), vjust = -0.5) +
  coord_cartesian(ylim = c(min(final_results$BIC) - 10, max(final_results$BIC) + 10)) +
  labs(title = "Model Selection by BIC", 
       subtitle = "Lower is Better", 
       x = "Model", 
       y = "BIC Score") +
  theme_minimal() +
  theme(legend.position = "none")

# Save and print plot
print(p_comp)
ggsave("final_comparison_bic.png", p_comp, width = 8, height = 6)