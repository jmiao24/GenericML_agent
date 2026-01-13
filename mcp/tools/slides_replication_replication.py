"""MCP tools for GenericML - Morocco microcredit study replication"""

import subprocess
import tempfile
from pathlib import Path
from typing import Annotated, Optional
import pandas as pd
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("genericml-slides_replication")

# Point to the R scripts DIRECTORY for this tutorial
R_SCRIPT_DIR = Path(__file__).parent.parent / "r_scripts" / "slides_replication_replication"


@mcp.tool()
def generic_ml(
    data_path: Annotated[str,
        "Path to .Rdata file containing Z (covariates matrix), D (treatment indicator), "
        "Y (outcome variable), vil_pair (village pair for fixed effects), and "
        "demi_paire (cluster variable for standard errors). "
        "Example: '/path/to/morocco_preprocessed.Rdata'"],
    learners: Annotated[str,
        "Comma-separated list of learner strings to use for estimation. "
        "Supported learners: 'random_forest', 'mlr3::lrn('cv_glmnet', s = 'lambda.min', alpha = 0.5)', "
        "'mlr3::lrn('svm')', 'mlr3::lrn('xgboost')'. "
        "Example: 'random_forest,mlr3::lrn('cv_glmnet', s = 'lambda.min', alpha = 0.5)'"],
    num_splits: Annotated[int,
        "Number of sample splits for cross-fitting procedure. "
        "Higher values provide more stable estimates but increase computation time. "
        "Default: 100"] = 100,
    quantile_cutoffs: Annotated[str,
        "Comma-separated quantile cutoffs for GATES grouping. "
        "Defines how observations are grouped by predicted treatment effects. "
        "Example: '0.2,0.4,0.6,0.8' creates 5 groups (quintiles). "
        "Default: '0.2,0.4,0.6,0.8'"] = "0.2,0.4,0.6,0.8",
    significance_level: Annotated[float,
        "Significance level for hypothesis tests and confidence intervals. "
        "Default: 0.05"] = 0.05,
    num_cores: Annotated[int,
        "Number of CPU cores for parallel computation. "
        "Must not exceed available cores on the system. "
        "Default: 2"] = 2,
    seed: Annotated[int,
        "Random number generator seed for reproducibility. "
        "Same seed with same data/parameters guarantees identical results. "
        "Default: 20220621"] = 20220621,
) -> dict:
    """
    Run GenericML analysis for heterogeneous treatment effect estimation.

    Fits machine learning models with sample splitting to estimate conditional average
    treatment effects (CATE), then prepares results for BLP, GATES, and CLAN analyses.
    This is the main workflow function that must be run before extraction tools.

    The function automatically:
    - Sets up BCA/CATE controls with fixed effects (vil_pair)
    - Configures clustered standard errors (demi_paire)
    - Runs cross-fitted ML estimation with specified learners
    - Saves GenericML object for subsequent analysis

    Returns RDS path needed for get_blp(), get_gates(), get_clan(), and get_best().
    """
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        output_csv = f.name

    rds_output = output_csv.replace(".csv", ".rds")

    subprocess.run([
        "Rscript", str(R_SCRIPT_DIR / "generic_ml.R"),
        "--data", data_path,
        "--learners", learners,
        "--num_splits", str(num_splits),
        "--quantile_cutoffs", quantile_cutoffs,
        "--significance_level", str(significance_level),
        "--num_cores", str(num_cores),
        "--seed", str(seed),
        "--output", output_csv,
        "--rds_output", rds_output
    ], check=True)

    result_df = pd.read_csv(output_csv)
    Path(output_csv).unlink()

    return {
        "message": f"GenericML analysis completed with {num_splits} splits and {len(learners.split(','))} learners",
        "reference": "https://github.com/mwelz/GenericML/blob/main/slides/replication/replication.R",
        "genml_rds_path": rds_output,
        "num_splits": int(result_df["num_splits"].iloc[0]),
        "num_learners": int(result_df["num_learners"].iloc[0]),
        "significance_level": float(result_df["significance_level"].iloc[0]),
    }


@mcp.tool()
def get_blp(
    genml_rds_path: Annotated[str,
        "Path to GenericML RDS file from generic_ml() output. "
        "This file contains the fitted GenericML object with all estimation results. "
        "Example: '/tmp/tmpxyz123.rds'"],
) -> dict:
    """
    Extract Best Linear Predictor (BLP) results from GenericML analysis.

    BLP provides a linear approximation of the conditional average treatment effect (CATE)
    using baseline covariates and the estimated CATE itself as predictors. It tests whether:
    - beta.1: Treatment effects vary with baseline covariates (heterogeneity)
    - beta.2: The ML proxy for CATE predicts actual treatment effects (validation)

    Returns coefficient estimates, confidence intervals, p-values, and a diagnostic plot.
    The plot visualizes the linear relationship between predicted and actual effects.
    """
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        output_csv = f.name

    subprocess.run([
        "Rscript", str(R_SCRIPT_DIR / "get_blp.R"),
        "--genml_rds", genml_rds_path,
        "--output", output_csv
    ], check=True)

    result_df = pd.read_csv(output_csv)
    plot_file = output_csv.replace(".csv", "_plot.png")
    Path(output_csv).unlink()

    return {
        "message": "BLP analysis completed",
        "reference": "https://github.com/mwelz/GenericML/blob/main/slides/replication/replication.R",
        "plot_path": plot_file,
        "results": result_df.to_dict(orient="records"),
    }


@mcp.tool()
def get_gates(
    genml_rds_path: Annotated[str,
        "Path to GenericML RDS file from generic_ml() output. "
        "This file contains the fitted GenericML object with all estimation results. "
        "Example: '/tmp/tmpxyz123.rds'"],
) -> dict:
    """
    Extract Group Average Treatment Effects (GATES) from GenericML analysis.

    GATES estimates average treatment effects within groups defined by the magnitude
    of predicted treatment effects. This reveals whether effects are concentrated
    in specific subpopulations:
    - gamma.1, gamma.2, ...: Average effect in each quantile group
    - gamma.5-gamma.1: Difference between highest and lowest groups (key test)

    Groups are defined by quantile_cutoffs specified in generic_ml().
    Returns estimates for each group, differences, and a visualization plot.
    """
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        output_csv = f.name

    subprocess.run([
        "Rscript", str(R_SCRIPT_DIR / "get_gates.R"),
        "--genml_rds", genml_rds_path,
        "--output", output_csv
    ], check=True)

    result_df = pd.read_csv(output_csv)
    plot_file = output_csv.replace(".csv", "_plot.png")
    Path(output_csv).unlink()

    return {
        "message": "GATES analysis completed",
        "reference": "https://github.com/mwelz/GenericML/blob/main/slides/replication/replication.R",
        "plot_path": plot_file,
        "results": result_df.to_dict(orient="records"),
    }


@mcp.tool()
def get_clan(
    genml_rds_path: Annotated[str,
        "Path to GenericML RDS file from generic_ml() output. "
        "This file contains the fitted GenericML object with all estimation results. "
        "Example: '/tmp/tmpxyz123.rds'"],
    variable: Annotated[str,
        "Name of baseline covariate to analyze for effect heterogeneity. "
        "Must be a column name present in the original data (Z matrix). "
        "CLAN reveals how treatment effects vary with this specific variable. "
        "Example: 'head_age_bl' (household head age at baseline)"],
) -> dict:
    """
    Perform Conditional Local Average treatment effect aNalysis (CLAN).

    CLAN examines how treatment effects vary with a specific baseline covariate,
    while controlling for confounding with other variables. Groups are formed by
    quantiles of the specified variable:
    - delta.1, delta.2, ...: Average effect in each quantile of the variable
    - delta.5-delta.1: Effect difference between highest and lowest quantiles

    This identifies whether the variable is a key driver of treatment effect
    heterogeneity. Returns estimates for each quantile group and a diagnostic plot.
    """
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        output_csv = f.name

    subprocess.run([
        "Rscript", str(R_SCRIPT_DIR / "get_clan.R"),
        "--genml_rds", genml_rds_path,
        "--variable", variable,
        "--output", output_csv
    ], check=True)

    result_df = pd.read_csv(output_csv)
    plot_file = output_csv.replace(".csv", "_plot.png")
    Path(output_csv).unlink()

    return {
        "message": f"CLAN analysis completed for variable '{variable}'",
        "reference": "https://github.com/mwelz/GenericML/blob/main/slides/replication/replication.R",
        "variable": variable,
        "plot_path": plot_file,
        "results": result_df.to_dict(orient="records"),
    }


@mcp.tool()
def get_best(
    genml_rds_path: Annotated[str,
        "Path to GenericML RDS file from generic_ml() output. "
        "This file contains the fitted GenericML object with all estimation results. "
        "Example: '/tmp/tmpxyz123.rds'"],
) -> dict:
    """
    Identify best-performing learner from GenericML analysis.

    Compares all learners based on mean squared error (MSE) of CATE predictions:
    - lambda: MSE for BLP estimation (lower is better)
    - lambda.bar: MSE for GATES/CLAN estimation (lower is better)

    Different learners may perform best for different analyses. The function
    reports MSE for all learners and identifies the optimal choice for each
    analysis type (BLP vs GATES/CLAN).

    Returns performance metrics for all learners and best learner selections.
    """
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        output_csv = f.name

    subprocess.run([
        "Rscript", str(R_SCRIPT_DIR / "get_best.R"),
        "--genml_rds", genml_rds_path,
        "--output", output_csv
    ], check=True)

    result_df = pd.read_csv(output_csv)
    Path(output_csv).unlink()

    # Split results into learner performance and best selections
    all_learners = result_df[result_df["type"] == "all_learners"].drop(columns=["type"]).to_dict(orient="records")
    best_selections = result_df[result_df["type"] == "best_selection"].drop(columns=["type"]).to_dict(orient="records")

    return {
        "message": "Best learner analysis completed",
        "reference": "https://github.com/mwelz/GenericML/blob/main/slides/replication/replication.R",
        "all_learners": all_learners,
        "best_selections": best_selections,
    }


if __name__ == "__main__":
    mcp.run()
