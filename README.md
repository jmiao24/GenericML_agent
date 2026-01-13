# [Paper2Agent](https://github.com/jmiao24/Paper2Agent): GenericML Demo

A demonstration of turning the [GenericML paper](https://arxiv.org/abs/1712.04802) into an interactive AI agent. This project transforms Chernozhukov, Demirer, Duflo and Fernández-Val's framework for generic machine learning inference on heterogeneous treatment effects into a conversational agent that can estimate treatment effect heterogeneity through natural language.

## Folder Structure

```
GenericML_Agent/
├── manuscript/
│   └── manuscript.md          # Full GenericML paper manuscript
├── mcp/
│   ├── GenericML_mcp.py       # MCP server entry point
│   ├── requirements.txt       # Python dependencies
│   ├── r_scripts/             # R scripts for each tool
│   │   ├── readme/            # Basic API functions
│   │   └── slides_replication_replication/  # Morocco study replication
│   └── tools/                 # Python wrappers for R scripts
└── README.md
```

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/jmiao24/GenericML_Agent.git
cd GenericML_Agent
```

### 2. Install Gemini CLI

Install the [Google Gemini CLI](https://github.com/google-gemini/gemini-cli):

```bash
brew install gemini-cli
```

### 3. Install R Dependencies

The agent requires R and the `GenericML` package. Install R from [CRAN](https://cran.r-project.org/), then install the required packages:

```r
install.packages("GenericML")
```

### 4. Install FastMCP

```bash
pip install fastmcp
```

### 5. Install MCP Server

Install the GenericML MCP server using fastmcp:

```bash
fastmcp install gemini-cli ./mcp/GenericML_mcp.py --with-requirements ./mcp/requirements.txt
```

### 6. Start the Agent

Start Gemini CLI in the repository folder:

```bash
gemini
```

You will now have access to the GenericML agent with all available tools.

## Example Query

```
Run GenericML analysis on my experimental data to estimate heterogeneous treatment
effects and identify which subgroups benefit most from the treatment.
```

## Available Agent Tools

The agent provides the following capabilities through natural language:

### Basic API (README)
- `readme_genericml`: Run Generic Machine Learning inference on heterogeneous treatment effects
- `readme_get_best`: Get the best learner for BLP, GATES, and CLAN analyses
- `readme_get_blp`: Get Best Linear Predictor (BLP) estimates of treatment effects
- `readme_get_gates`: Get Grouped Average Treatment Effects (GATES) estimates
- `readme_get_clan`: Get Classification Analysis (CLAN) estimates for a variable

### Morocco Microcredit Study
- `morocco_generic_ml`: Run GenericML analysis for heterogeneous treatment effect estimation
- `morocco_get_blp`: Extract Best Linear Predictor (BLP) results from GenericML analysis
- `morocco_get_gates`: Extract Group Average Treatment Effects (GATES) from GenericML analysis
- `morocco_get_clan`: Perform Conditional Local Average treatment effect aNalysis (CLAN)
- `morocco_get_best`: Identify best-performing learner from GenericML analysis

## About GenericML

GenericML provides methods for generic machine learning inference on heterogeneous treatment effects in randomized experiments. Key features include:

- **Best Linear Predictor (BLP)**: Linear projection of treatment effects on ML proxy predictors
- **Grouped Average Treatment Effects (GATES)**: Average effects sorted by impact groups
- **Classification Analysis (CLAN)**: Average characteristics of most and least impacted units
- Works with any ML method (penalized regression, random forests, neural networks, boosted trees, ensembles)
- Quantile aggregation across multiple data splits for robust inference
- Valid inference in high-dimensional settings

For more details, see the manuscript in `manuscript/manuscript.md`.
