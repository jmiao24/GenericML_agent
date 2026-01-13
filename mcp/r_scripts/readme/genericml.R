#!/usr/bin/env Rscript
library(optparse)
library(data.table)
library(GenericML)

option_list <- list(
  make_option("--Z", type = "character", help = "Covariates CSV file (N x p matrix)"),
  make_option("--D", type = "character", help = "Treatment assignment CSV file (single column, 0/1)"),
  make_option("--Y", type = "character", help = "Outcomes CSV file (single column)"),
  make_option("--learners_GenericML", type = "character", help = "Comma-separated learner specifications"),
  make_option("--learner_propensity_score", type = "character", default = "constant", help = "Propensity score learner [default: %default]"),
  make_option("--num_splits", type = "integer", default = 100, help = "Number of sample splits [default: %default]"),
  make_option("--Z_CLAN", type = "character", default = "", help = "Optional CLAN variables CSV file"),
  make_option("--HT", type = "logical", default = FALSE, help = "Use Horvitz-Thompson transformation [default: %default]"),
  make_option("--quantile_cutoffs", type = "character", default = "0.25,0.5,0.75", help = "GATES quantile cutoffs [default: %default]"),
  make_option("--prop_aux", type = "numeric", default = 0.5, help = "Proportion of auxiliary samples [default: %default]"),
  make_option("--significance_level", type = "numeric", default = 0.05, help = "Significance level [default: %default]"),
  make_option("--parallel", type = "logical", default = FALSE, help = "Use parallel computing [default: %default]"),
  make_option("--num_cores", type = "integer", default = 1, help = "Number of cores [default: %default]"),
  make_option("--seed", type = "integer", default = NULL, help = "Random seed"),
  make_option("--output", type = "character", help = "Output RDS file (required)")
)

args <- parse_args(OptionParser(option_list = option_list))

# Read inputs
Z <- as.matrix(fread(args$Z))
D <- fread(args$D)[[1]]
Y <- fread(args$Y)[[1]]

# Parse learners
learners <- strsplit(args$learners_GenericML, ",")[[1]]

# Parse quantile cutoffs
quantile_cutoffs <- as.numeric(strsplit(args$quantile_cutoffs, ",")[[1]])

# Read Z_CLAN if provided
Z_CLAN <- if (args$Z_CLAN != "") as.matrix(fread(args$Z_CLAN)) else NULL

# Run GenericML
genML <- GenericML(
  Z = Z,
  D = D,
  Y = Y,
  learners_GenericML = learners,
  learner_propensity_score = args$learner_propensity_score,
  num_splits = args$num_splits,
  Z_CLAN = Z_CLAN,
  HT = args$HT,
  quantile_cutoffs = quantile_cutoffs,
  prop_aux = args$prop_aux,
  significance_level = args$significance_level,
  parallel = args$parallel,
  num_cores = args$num_cores,
  seed = args$seed,
  store_splits = TRUE,
  store_learners = FALSE
)

# Save GenericML object
saveRDS(genML, args$output)
