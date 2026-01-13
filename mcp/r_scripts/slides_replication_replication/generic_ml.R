#!/usr/bin/env Rscript
library(optparse)
library(data.table)
library(GenericML)

option_list <- list(
  make_option("--data", type = "character", help = "Path to .Rdata file containing Z, D, Y, vil_pair, demi_paire"),
  make_option("--learners", type = "character", help = "Comma-separated list of learner strings"),
  make_option("--num_splits", type = "integer", default = 100, help = "Number of splits [default: %default]"),
  make_option("--quantile_cutoffs", type = "character", default = "0.2,0.4,0.6,0.8", help = "Comma-separated quantile cutoffs [default: %default]"),
  make_option("--significance_level", type = "double", default = 0.05, help = "Significance level [default: %default]"),
  make_option("--num_cores", type = "integer", default = 2, help = "Number of cores for parallelization [default: %default]"),
  make_option("--seed", type = "integer", default = 20220621, help = "RNG seed [default: %default]"),
  make_option("--output", type = "character", help = "Output CSV file for summary (required)"),
  make_option("--rds_output", type = "character", help = "Output RDS file for GenericML object (required)")
)

args <- parse_args(OptionParser(option_list = option_list))

# Load data
load(args$data)

# Parse learners - use regex to split on commas NOT inside parentheses
# This handles mlr3::lrn() calls that contain commas in their arguments
learners <- strsplit(args$learners, ",(?=\\s*(?:random_forest|lasso|tree|mlr3))", perl=TRUE)[[1]]

# Parse quantile cutoffs
quantile_cutoffs <- as.numeric(strsplit(args$quantile_cutoffs, ",")[[1]])

# Setup X1 and vcov
X1 <- setup_X1(funs_Z = c("B", "S"), fixed_effects = vil_pair)
vcov <- setup_vcov(estimator = "vcovCL", arguments = list(cluster = demi_paire))

# Run GenericML
genML <- GenericML(
  Z = Z, D = D, Y = Y,
  learners_GenericML = learners,
  learner_propensity_score = "constant",
  num_splits = args$num_splits,
  quantile_cutoffs = quantile_cutoffs,
  significance_level = args$significance_level,
  X1_BLP = X1, X1_GATES = X1,
  vcov_BLP = vcov, vcov_GATES = vcov,
  parallel = TRUE, num_cores = args$num_cores,
  seed = args$seed
)

# Save GenericML object
saveRDS(genML, args$rds_output)

# Create summary output
summary_data <- data.table(
  num_splits = args$num_splits,
  num_learners = length(learners),
  significance_level = args$significance_level,
  rds_path = args$rds_output
)
fwrite(summary_data, args$output)
