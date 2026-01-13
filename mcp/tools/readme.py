"""MCP tools for GenericML - Generic Machine Learning Inference

This module provides tools for conducting generic machine learning inference on
heterogeneous treatment effects in randomized experiments as described in
Chernozhukov, Demirer, Duflo and Fernández-Val (2020).
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Annotated, Optional
import pandas as pd
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("genericml-readme")

R_SCRIPT_DIR = Path(__file__).parent.parent / "r_scripts" / "readme"


@mcp.tool()
def genericml(
    Z_csv: Annotated[str,
        "Path to CSV file with covariates (N × p matrix). "
        "Each row is one observation, columns are features."],
    D_csv: Annotated[str,
        "Path to CSV file with binary treatment assignment (single column, 0/1). "
        "Value 1 denotes treatment group, 0 denotes control group."],
    Y_csv: Annotated[str,
        "Path to CSV file with outcomes (single column)."],
    learners_GenericML: Annotated[str,
        "Comma-separated machine learning methods for BCA and CATE estimation. "
        "Options: 'lasso', 'random_forest', 'tree', or mlr3 syntax. "
        "Example: 'lasso,mlr3::lrn(ranger, num.trees = 100)'"],
    learner_propensity_score: Annotated[str,
        "Propensity score learner specification. "
        "Can be 'constant' (default), 'lasso', 'random_forest', 'tree', or mlr3 syntax. "
        "Example: 'mlr3::lrn(glmnet, lambda = 0, alpha = 1)'"] = "constant",
    num_splits: Annotated[int,
        "Number of sample splits for cross-fitting. Must be > 1."] = 100,
    Z_CLAN_csv: Annotated[Optional[str],
        "Path to CSV file with variables for CLAN (Classification Analysis). "
        "If not provided, uses Z."] = None,
    HT: Annotated[bool,
        "Whether to use Horvitz-Thompson transformation in BLP and GATES regressions."] = False,
    quantile_cutoffs: Annotated[str,
        "Comma-separated quantile cutoffs for GATES grouping. "
        "Example: '0.25,0.5,0.75' for quartiles."] = "0.25,0.5,0.75",
    prop_aux: Annotated[float,
        "Proportion of samples in auxiliary set (0 < prop_aux < 1)."] = 0.5,
    significance_level: Annotated[float,
        "Significance level for inference."] = 0.05,
    parallel: Annotated[bool,
        "Whether to use parallel computing."] = False,
    num_cores: Annotated[int,
        "Number of cores for parallel computing."] = 1,
    seed: Annotated[Optional[int],
        "Random seed for reproducibility."] = None,
) -> dict:
    """Run Generic Machine Learning inference on heterogeneous treatment effects.

    This function implements the generic ML inference framework for analyzing
    heterogeneous treatment effects in randomized experiments. It estimates:
    - BLP (Best Linear Predictor) of treatment effects
    - GATES (Grouped Average Treatment Effects)
    - CLAN (Classification Analysis) of heterogeneity sources

    Returns a GenericML object saved as RDS file that can be used with
    get_best, get_blp, get_gates, and get_clan accessor functions.
    """
    with tempfile.NamedTemporaryFile(suffix=".rds", delete=False) as f:
        output_rds = f.name

    cmd = [
        "Rscript", str(R_SCRIPT_DIR / "genericml.R"),
        "--Z", Z_csv,
        "--D", D_csv,
        "--Y", Y_csv,
        "--learners_GenericML", learners_GenericML,
        "--learner_propensity_score", learner_propensity_score,
        "--num_splits", str(num_splits),
        "--quantile_cutoffs", quantile_cutoffs,
        "--prop_aux", str(prop_aux),
        "--significance_level", str(significance_level),
        "--num_cores", str(num_cores),
        "--output", output_rds
    ]

    if Z_CLAN_csv:
        cmd.extend(["--Z_CLAN", Z_CLAN_csv])
    else:
        cmd.extend(["--Z_CLAN", ""])

    if HT:
        cmd.extend(["--HT", "TRUE"])

    if parallel:
        cmd.extend(["--parallel", "TRUE"])

    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    subprocess.run(cmd, check=True)

    return {
        "message": f"GenericML analysis completed with {num_splits} splits",
        "reference": "https://github.com/mwelz/GenericML/blob/main/README.md",
        "genericml_path": output_rds,
        "note": "Use get_best(), get_blp(), get_gates(), and get_clan() to extract results"
    }


@mcp.tool()
def get_best(
    genericml_rds: Annotated[str,
        "Path to GenericML RDS file (from genericml output)."],
) -> dict:
    """Get the best learner for BLP, GATES, and CLAN analyses.

    The best learner is determined by maximizing the Lambda criteria.
    Returns which learner performs best for each type of analysis along
    with the Lambda and Lambda-bar performance metrics.
    """
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        output_csv = f.name

    subprocess.run([
        "Rscript", str(R_SCRIPT_DIR / "get_best.R"),
        "--genericml", genericml_rds,
        "--output", output_csv
    ], check=True)

    result_df = pd.read_csv(output_csv)
    overview_csv = output_csv.replace(".csv", "_overview.csv")
    overview_df = pd.read_csv(overview_csv)

    Path(output_csv).unlink()
    Path(overview_csv).unlink()

    return {
        "message": "Best learner information retrieved",
        "reference": "https://github.com/mwelz/GenericML/blob/main/README.md",
        "best_learners": result_df.to_dict(orient="records"),
        "overview": overview_df.to_dict(orient="records"),
    }


@mcp.tool()
def get_blp(
    genericml_rds: Annotated[str,
        "Path to GenericML RDS file (from genericml output)."],
    learner: Annotated[str,
        "Learner to use. Use 'best' for the best learner, or specify a learner name."] = "best",
) -> dict:
    """Get Best Linear Predictor (BLP) estimates of treatment effects.

    BLP provides a linear approximation to the conditional average treatment
    effect (CATE) function, with inference using the variance inflation method.
    Returns point estimates, confidence intervals, and p-values for BLP parameters
    along with a visualization.
    """
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        output_csv = f.name

    subprocess.run([
        "Rscript", str(R_SCRIPT_DIR / "get_blp.R"),
        "--genericml", genericml_rds,
        "--learner", learner,
        "--output", output_csv
    ], check=True)

    result_df = pd.read_csv(output_csv)
    plot_file = output_csv.replace(".csv", "_plot.png")
    Path(output_csv).unlink()

    return {
        "message": f"BLP estimates retrieved for learner: {learner}",
        "reference": "https://github.com/mwelz/GenericML/blob/main/README.md",
        "results": result_df.to_dict(orient="records"),
        "plot_path": plot_file if Path(plot_file).exists() else None,
    }


@mcp.tool()
def get_gates(
    genericml_rds: Annotated[str,
        "Path to GenericML RDS file (from genericml output)."],
    learner: Annotated[str,
        "Learner to use. Use 'best' for the best learner, or specify a learner name."] = "best",
) -> dict:
    """Get Grouped Average Treatment Effects (GATES) estimates.

    GATES partitions observations into groups based on predicted CATE and
    estimates average treatment effects within each group. This reveals
    treatment effect heterogeneity patterns. Returns estimates, confidence
    intervals, and p-values for each group along with a visualization.
    """
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        output_csv = f.name

    subprocess.run([
        "Rscript", str(R_SCRIPT_DIR / "get_gates.R"),
        "--genericml", genericml_rds,
        "--learner", learner,
        "--output", output_csv
    ], check=True)

    result_df = pd.read_csv(output_csv)
    plot_file = output_csv.replace(".csv", "_plot.png")
    Path(output_csv).unlink()

    return {
        "message": f"GATES estimates retrieved for learner: {learner}",
        "reference": "https://github.com/mwelz/GenericML/blob/main/README.md",
        "results": result_df.to_dict(orient="records"),
        "plot_path": plot_file if Path(plot_file).exists() else None,
    }


@mcp.tool()
def get_clan(
    genericml_rds: Annotated[str,
        "Path to GenericML RDS file (from genericml output)."],
    variable: Annotated[str,
        "Name of the variable for CLAN analysis (must match column name in Z_CLAN)."],
    learner: Annotated[str,
        "Learner to use. Use 'best' for the best learner, or specify a learner name."] = "best",
) -> dict:
    """Get Classification Analysis (CLAN) estimates for a variable.

    CLAN identifies which covariates are associated with treatment effect
    heterogeneity by grouping observations based on a covariate and comparing
    treatment effects across groups. Returns estimates, confidence intervals,
    and p-values along with a visualization showing the relationship between
    the variable and treatment effects.
    """
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        output_csv = f.name

    subprocess.run([
        "Rscript", str(R_SCRIPT_DIR / "get_clan.R"),
        "--genericml", genericml_rds,
        "--variable", variable,
        "--learner", learner,
        "--output", output_csv
    ], check=True)

    result_df = pd.read_csv(output_csv)
    plot_file = output_csv.replace(".csv", "_plot.png")
    Path(output_csv).unlink()

    return {
        "message": f"CLAN estimates retrieved for variable: {variable}, learner: {learner}",
        "reference": "https://github.com/mwelz/GenericML/blob/main/README.md",
        "results": result_df.to_dict(orient="records"),
        "plot_path": plot_file if Path(plot_file).exists() else None,
    }


if __name__ == "__main__":
    mcp.run()
