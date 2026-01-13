"""
Model Context Protocol (MCP) for GenericML

GenericML provides methods for generic machine learning inference on heterogeneous
treatment effects in randomized experiments, implementing the framework described in
Chernozhukov, Demirer, Duflo and Fernández-Val (2020).

This MCP Server provides Python interfaces to R tools extracted from the following tutorial files:
1. README.md (basic API)
    - readme_genericml: Run Generic Machine Learning inference on heterogeneous treatment effects (calls R via Rscript)
    - readme_get_best: Get the best learner for BLP, GATES, and CLAN analyses (calls R via Rscript)
    - readme_get_blp: Get Best Linear Predictor (BLP) estimates of treatment effects (calls R via Rscript)
    - readme_get_gates: Get Grouped Average Treatment Effects (GATES) estimates (calls R via Rscript)
    - readme_get_clan: Get Classification Analysis (CLAN) estimates for a variable (calls R via Rscript)
2. slides/replication/replication.R (Morocco microcredit study)
    - morocco_generic_ml: Run GenericML analysis for heterogeneous treatment effect estimation (calls R via Rscript)
    - morocco_get_blp: Extract Best Linear Predictor (BLP) results from GenericML analysis (calls R via Rscript)
    - morocco_get_gates: Extract Group Average Treatment Effects (GATES) from GenericML analysis (calls R via Rscript)
    - morocco_get_clan: Perform Conditional Local Average treatment effect aNalysis (CLAN) (calls R via Rscript)
    - morocco_get_best: Identify best-performing learner from GenericML analysis (calls R via Rscript)

Note: All tools execute R code via Rscript subprocess calls. Ensure R is installed
and the package dependencies are available in the renv environment at repo/GenericML/.
"""

from fastmcp import FastMCP

# Import tool functions from modules (alphabetical order)
from tools import readme
from tools import slides_replication_replication

# Server definition
mcp = FastMCP(name="GenericML")

# Register tools from readme module (basic API)
mcp.tool(name="readme_genericml")(readme.genericml)
mcp.tool(name="readme_get_best")(readme.get_best)
mcp.tool(name="readme_get_blp")(readme.get_blp)
mcp.tool(name="readme_get_gates")(readme.get_gates)
mcp.tool(name="readme_get_clan")(readme.get_clan)

# Register tools from slides_replication_replication module (Morocco study)
mcp.tool(name="morocco_generic_ml")(slides_replication_replication.generic_ml)
mcp.tool(name="morocco_get_blp")(slides_replication_replication.get_blp)
mcp.tool(name="morocco_get_gates")(slides_replication_replication.get_gates)
mcp.tool(name="morocco_get_clan")(slides_replication_replication.get_clan)
mcp.tool(name="morocco_get_best")(slides_replication_replication.get_best)

if __name__ == "__main__":
    mcp.run()
