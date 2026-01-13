#!/usr/bin/env Rscript
library(optparse)
library(data.table)
library(ragg)
library(GenericML)

option_list <- list(
  make_option("--genericml", type = "character", help = "Path to GenericML RDS file"),
  make_option("--learner", type = "character", default = "best", help = "Learner to use [default: %default]"),
  make_option("--output", type = "character", help = "Output CSV file (required)")
)

args <- parse_args(OptionParser(option_list = option_list))

genML <- readRDS(args$genericml)
blp <- get_BLP(genML, learner = args$learner, plot = TRUE)

# Save plot
plot_file <- sub("\\.csv$", "_plot.png", args$output)
ragg::agg_png(plot_file, width = 800, height = 600, res = 150)
print(blp$plot)
dev.off()

# Prepare output data
result <- data.table(
  parameter = names(blp$estimate),
  estimate = blp$estimate,
  ci_lower = blp$confidence_interval[, "lower"],
  ci_upper = blp$confidence_interval[, "upper"],
  p_value = blp$p_value,
  confidence_level = blp$confidence_level
)

fwrite(result, args$output)
