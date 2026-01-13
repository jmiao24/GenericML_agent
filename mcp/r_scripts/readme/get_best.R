#!/usr/bin/env Rscript
library(optparse)
library(data.table)
library(GenericML)

option_list <- list(
  make_option("--genericml", type = "character", help = "Path to GenericML RDS file"),
  make_option("--output", type = "character", help = "Output CSV file (required)")
)

args <- parse_args(OptionParser(option_list = option_list))

genML <- readRDS(args$genericml)
best <- get_best(genML)

# Prepare output data
result <- data.table(
  analysis = c("BLP", "GATES", "CLAN"),
  best_learner = c(best$BLP, best$GATES, best$CLAN)
)

# Add overview table
overview_dt <- as.data.table(best$overview, keep.rownames = "learner")
overview_file <- sub("\\.csv$", "_overview.csv", args$output)
fwrite(overview_dt, overview_file)

fwrite(result, args$output)
