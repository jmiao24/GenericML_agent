#!/usr/bin/env Rscript
library(optparse)
library(data.table)
library(ragg)
library(GenericML)

option_list <- list(
  make_option("--genml_rds", type = "character", help = "Path to GenericML RDS file"),
  make_option("--variable", type = "character", help = "Variable name for CLAN analysis"),
  make_option("--output", type = "character", help = "Output CSV file (required)")
)

args <- parse_args(OptionParser(option_list = option_list))

# Load GenericML object
genML <- readRDS(args$genml_rds)

# Get CLAN results with plot
results_CLAN <- get_CLAN(genML, variable = args$variable, plot = TRUE)

# Save plot
plot_file <- sub("\\.csv$", "_plot.png", args$output)
ragg::agg_png(plot_file, width = 800, height = 600, res = 150)
plot(results_CLAN)
dev.off()

# Extract results table from CLAN components
result_dt <- data.table(
  coefficient = names(results_CLAN$estimate),
  estimate = as.numeric(results_CLAN$estimate),
  ci_lower = as.numeric(results_CLAN$confidence_interval[, "lower"]),
  ci_upper = as.numeric(results_CLAN$confidence_interval[, "upper"]),
  pvalue = as.numeric(results_CLAN$p_value)
)

fwrite(result_dt, args$output)
