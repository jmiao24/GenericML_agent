#!/usr/bin/env Rscript
library(optparse)
library(data.table)
library(ragg)
library(GenericML)

option_list <- list(
  make_option("--genericml", type = "character", help = "Path to GenericML RDS file"),
  make_option("--variable", type = "character", help = "CLAN variable name (required)"),
  make_option("--learner", type = "character", default = "best", help = "Learner to use [default: %default]"),
  make_option("--output", type = "character", help = "Output CSV file (required)")
)

args <- parse_args(OptionParser(option_list = option_list))

genML <- readRDS(args$genericml)
clan <- get_CLAN(genML, variable = args$variable, learner = args$learner, plot = TRUE)

# Save plot
plot_file <- sub("\\.csv$", "_plot.png", args$output)
ragg::agg_png(plot_file, width = 800, height = 600, res = 150)
print(clan$plot)
dev.off()

# Prepare output data
result <- data.table(
  parameter = names(clan$estimate),
  estimate = clan$estimate,
  ci_lower = clan$confidence_interval[, "lower"],
  ci_upper = clan$confidence_interval[, "upper"],
  p_value = clan$p_value,
  confidence_level = clan$confidence_level,
  variable = clan$CLAN_variable
)

fwrite(result, args$output)
