#!/usr/bin/env Rscript
library(optparse)
library(data.table)
library(GenericML)

option_list <- list(
  make_option("--genml_rds", type = "character", help = "Path to GenericML RDS file"),
  make_option("--output", type = "character", help = "Output CSV file (required)")
)

args <- parse_args(OptionParser(option_list = option_list))

# Load GenericML object
genML <- readRDS(args$genml_rds)

# Get best learner results
results_best <- get_best(genML)

# Extract overview table with all learners
overview_matrix <- results_best$overview
result_dt <- data.table(
  learner = rownames(overview_matrix),
  lambda = as.numeric(overview_matrix[, "lambda"]),
  lambda.bar = as.numeric(overview_matrix[, "lambda.bar"]),
  type = "all_learners"
)

# Add best learner selections
best_info <- data.table(
  learner = c(results_best$BLP, results_best$GATES),
  lambda = c(NA_real_, NA_real_),
  lambda.bar = c(NA_real_, NA_real_),
  type = "best_selection",
  metric = c("BLP", "GATES/CLAN")
)

combined <- rbindlist(list(result_dt, best_info), fill = TRUE)
fwrite(combined, args$output)
