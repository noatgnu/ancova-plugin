import os

import click
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def run_ancova_single(
    data: pd.DataFrame,
    dependent_var: str,
    factor: str,
    covariates: list[str],
    ss_type: int = 2
) -> dict:
    """
    Run ANCOVA for a single dependent variable.

    :param data: DataFrame with all variables.
    :param dependent_var: Name of dependent variable column.
    :param factor: Name of the categorical factor column.
    :param covariates: List of covariate column names.
    :param ss_type: Type of sum of squares (1, 2, or 3).
    :return: Dictionary with ANCOVA results.
    """
    data_clean = data[[dependent_var, factor] + covariates].dropna()

    if len(data_clean) < 3:
        return {
            "feature": dependent_var,
            "n": len(data_clean),
            "factor_f": np.nan,
            "factor_p": np.nan,
            "factor_eta_sq": np.nan,
            "residual_std": np.nan,
            "r_squared": np.nan,
            "error": "Insufficient data"
        }

    n_groups = data_clean[factor].nunique()
    if n_groups < 2:
        return {
            "feature": dependent_var,
            "n": len(data_clean),
            "factor_f": np.nan,
            "factor_p": np.nan,
            "factor_eta_sq": np.nan,
            "residual_std": np.nan,
            "r_squared": np.nan,
            "error": "Need at least 2 groups"
        }

    cov_terms = []
    for cov in covariates:
        if data_clean[cov].dtype == 'object' or data_clean[cov].nunique() < 10:
            cov_terms.append(f"C({cov})")
        else:
            cov_terms.append(cov)

    covariate_formula = " + ".join(cov_terms) if cov_terms else ""
    formula = f"Q('{dependent_var}') ~ C({factor})"
    if covariate_formula:
        formula += f" + {covariate_formula}"

    try:
        model = ols(formula, data=data_clean).fit()
        anova_table = anova_lm(model, typ=ss_type)

        factor_key = f"C({factor})"
        if factor_key not in anova_table.index:
            for idx in anova_table.index:
                if factor in str(idx):
                    factor_key = idx
                    break

        factor_f = anova_table.loc[factor_key, "F"] if factor_key in anova_table.index else np.nan
        factor_p = anova_table.loc[factor_key, "PR(>F)"] if factor_key in anova_table.index else np.nan

        ss_total = anova_table["sum_sq"].sum()
        factor_ss = anova_table.loc[factor_key, "sum_sq"] if factor_key in anova_table.index else 0
        factor_eta_sq = factor_ss / ss_total if ss_total > 0 else np.nan

        return {
            "feature": dependent_var,
            "n": len(data_clean),
            "factor_f": factor_f,
            "factor_p": factor_p,
            "factor_eta_sq": factor_eta_sq,
            "residual_std": np.sqrt(model.mse_resid),
            "r_squared": model.rsquared,
            "error": None
        }

    except Exception as e:
        return {
            "feature": dependent_var,
            "n": len(data_clean),
            "factor_f": np.nan,
            "factor_p": np.nan,
            "factor_eta_sq": np.nan,
            "residual_std": np.nan,
            "r_squared": np.nan,
            "error": str(e)
        }


def compute_adjusted_means(
    data: pd.DataFrame,
    dependent_var: str,
    factor: str,
    covariates: list[str]
) -> pd.DataFrame:
    """
    Compute covariate-adjusted means for each factor level.

    :param data: DataFrame with all variables.
    :param dependent_var: Name of dependent variable column.
    :param factor: Name of the categorical factor column.
    :param covariates: List of covariate column names.
    :return: DataFrame with adjusted means.
    """
    data_clean = data[[dependent_var, factor] + covariates].dropna()

    if len(data_clean) < 3:
        return pd.DataFrame()

    cov_terms = []
    for cov in covariates:
        if data_clean[cov].dtype == 'object' or data_clean[cov].nunique() < 10:
            cov_terms.append(f"C({cov})")
        else:
            cov_terms.append(cov)

    covariate_formula = " + ".join(cov_terms) if cov_terms else ""
    formula = f"Q('{dependent_var}') ~ C({factor})"
    if covariate_formula:
        formula += f" + {covariate_formula}"

    try:
        model = ols(formula, data=data_clean).fit()

        covariate_means = {}
        for cov in covariates:
            if data_clean[cov].dtype in ['float64', 'int64', 'float32', 'int32']:
                covariate_means[cov] = data_clean[cov].mean()

        adjusted_means = []
        for level in data_clean[factor].unique():
            pred_data = {factor: [level]}
            for cov in covariates:
                if cov in covariate_means:
                    pred_data[cov] = [covariate_means[cov]]
                else:
                    mode_val = data_clean[cov].mode()
                    pred_data[cov] = [mode_val.iloc[0] if len(mode_val) > 0 else data_clean[cov].iloc[0]]

            pred_df = pd.DataFrame(pred_data)
            adj_mean = model.predict(pred_df)[0]
            adjusted_means.append({
                "factor_level": level,
                "adjusted_mean": adj_mean,
                "n": len(data_clean[data_clean[factor] == level])
            })

        return pd.DataFrame(adjusted_means)

    except Exception:
        return pd.DataFrame()


def generate_volcano_plot(results_df: pd.DataFrame, output_folder: str):
    """
    Generate volcano plot of ANCOVA results.

    :param results_df: DataFrame with ANCOVA results.
    :param output_folder: Output folder path.
    """
    plot_df = results_df.dropna(subset=["factor_p", "factor_eta_sq"])
    if len(plot_df) == 0:
        return

    plot_df = plot_df.copy()
    plot_df["-log10(p)"] = -np.log10(plot_df["factor_p"].clip(lower=1e-300))
    plot_df["-log10(adj_p)"] = -np.log10(plot_df["adj_p"].clip(lower=1e-300))
    plot_df["significant"] = plot_df["adj_p"] < 0.05

    fig = px.scatter(
        plot_df,
        x="factor_eta_sq",
        y="-log10(adj_p)",
        color="significant",
        hover_name="feature",
        hover_data=["factor_f", "factor_p", "adj_p", "n"],
        title="ANCOVA Volcano Plot",
        labels={
            "factor_eta_sq": "Effect Size (Eta-squared)",
            "-log10(adj_p)": "-log10(Adjusted P-value)"
        },
        color_discrete_map={True: "red", False: "gray"}
    )

    fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="blue",
                  annotation_text="FDR = 0.05")

    fig.update_layout(template="plotly_white", width=900, height=600)
    fig.write_html(os.path.join(output_folder, "volcano_plot.html"))


def generate_effect_size_plot(results_df: pd.DataFrame, output_folder: str, top_n: int = 30):
    """
    Generate bar plot of top effect sizes.

    :param results_df: DataFrame with ANCOVA results.
    :param output_folder: Output folder path.
    :param top_n: Number of top features to show.
    """
    plot_df = results_df.dropna(subset=["factor_eta_sq", "adj_p"])
    plot_df = plot_df.nsmallest(top_n, "adj_p")

    if len(plot_df) == 0:
        return

    plot_df = plot_df.sort_values("factor_eta_sq", ascending=True)

    fig = px.bar(
        plot_df,
        x="factor_eta_sq",
        y="feature",
        orientation="h",
        color="adj_p",
        color_continuous_scale="Viridis_r",
        title=f"Top {min(top_n, len(plot_df))} Features by Effect Size",
        labels={
            "factor_eta_sq": "Effect Size (Eta-squared)",
            "feature": "Feature",
            "adj_p": "Adjusted P-value"
        }
    )

    fig.update_layout(template="plotly_white", height=max(400, len(plot_df) * 25))
    fig.write_html(os.path.join(output_folder, "effect_size_plot.html"))


def generate_pvalue_histogram(results_df: pd.DataFrame, output_folder: str):
    """
    Generate p-value distribution histogram.

    :param results_df: DataFrame with ANCOVA results.
    :param output_folder: Output folder path.
    """
    plot_df = results_df.dropna(subset=["factor_p"])

    if len(plot_df) == 0:
        return

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Raw P-values", "Adjusted P-values (BH)"))

    fig.add_trace(
        go.Histogram(x=plot_df["factor_p"], nbinsx=50, name="Raw"),
        row=1, col=1
    )

    fig.add_trace(
        go.Histogram(x=plot_df["adj_p"], nbinsx=50, name="Adjusted"),
        row=1, col=2
    )

    fig.update_layout(
        title="P-value Distribution",
        template="plotly_white",
        showlegend=False,
        width=900,
        height=400
    )

    fig.update_xaxes(title_text="P-value", row=1, col=1)
    fig.update_xaxes(title_text="Adjusted P-value (BH)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)

    fig.write_html(os.path.join(output_folder, "pvalue_histogram.html"))


def ancova_batch(
    input_file: str,
    annotation_file: str,
    output_folder: str,
    sample_cols: list[str],
    factor_col: str,
    covariate_cols: list[str],
    index_col: str = None,
    ss_type: int = 2,
    alpha: float = 0.05,
    log2_transform: bool = False
):
    """
    Run batch ANCOVA analysis on multiple features.

    :param input_file: Path to input data file (features as rows, samples as columns).
    :param annotation_file: Path to annotation file (Sample, Condition, covariates).
    :param output_folder: Path to output folder.
    :param sample_cols: List of sample column names to analyze.
    :param factor_col: Name of the main factor column in annotation file.
    :param covariate_cols: List of covariate column names in annotation file.
    :param index_col: Name of index column for feature IDs.
    :param ss_type: Type of sum of squares (1, 2, or 3).
    :param alpha: Significance threshold for FDR.
    :param log2_transform: Whether to log2 transform the data.
    """
    if input_file.endswith(".tsv") or input_file.endswith(".txt"):
        df = pd.read_csv(input_file, sep="\t")
    elif input_file.endswith(".csv"):
        df = pd.read_csv(input_file, sep=",")
    else:
        raise ValueError("Unsupported file format for input file")

    if annotation_file.endswith(".tsv") or annotation_file.endswith(".txt"):
        annotation_df = pd.read_csv(annotation_file, sep="\t")
    elif annotation_file.endswith(".csv"):
        annotation_df = pd.read_csv(annotation_file, sep=",")
    else:
        raise ValueError("Unsupported file format for annotation file")

    os.makedirs(output_folder, exist_ok=True)

    sample_col_name = "Sample"
    if sample_col_name not in annotation_df.columns:
        sample_col_name = annotation_df.columns[0]

    if factor_col not in annotation_df.columns:
        raise ValueError(f"Factor column '{factor_col}' not found in annotation file")

    for cov in covariate_cols:
        if cov not in annotation_df.columns:
            raise ValueError(f"Covariate column '{cov}' not found in annotation file")

    if index_col and index_col in df.columns:
        feature_names = df[index_col].tolist()
    else:
        feature_names = [f"Feature_{i}" for i in range(len(df))]

    valid_samples = [s for s in sample_cols if s in df.columns]
    if not valid_samples:
        raise ValueError("No valid sample columns found in data file")

    print(f"[INFO] Found {len(valid_samples)} sample columns")
    print(f"[INFO] Found {len(feature_names)} features")
    print(f"[INFO] Factor: {factor_col}")
    print(f"[INFO] Covariates: {covariate_cols}")

    data_long_records = []
    for i, row in df.iterrows():
        feature_name = feature_names[i]
        for sample in valid_samples:
            value = row[sample]
            if log2_transform and pd.notna(value) and value > 0:
                value = np.log2(value)
            data_long_records.append({
                "feature": feature_name,
                sample_col_name: sample,
                "value": value
            })

    data_long = pd.DataFrame(data_long_records)

    annotation_subset = annotation_df[[sample_col_name, factor_col] + covariate_cols].copy()

    print(f"[1/4] Running ANCOVA for {len(feature_names)} features...")

    results = []
    for idx, feature in enumerate(feature_names):
        if (idx + 1) % 100 == 0:
            print(f"    Processing feature {idx + 1}/{len(feature_names)}...")

        feature_data = data_long[data_long["feature"] == feature][[sample_col_name, "value"]]
        feature_data = feature_data.merge(annotation_subset, on=sample_col_name, how="inner")

        if len(feature_data) < 3:
            results.append({
                "feature": feature,
                "n": len(feature_data),
                "factor_f": np.nan,
                "factor_p": np.nan,
                "factor_eta_sq": np.nan,
                "residual_std": np.nan,
                "r_squared": np.nan,
                "error": "Insufficient data after merge"
            })
            continue

        result = run_ancova_single(
            data=feature_data,
            dependent_var="value",
            factor=factor_col,
            covariates=covariate_cols,
            ss_type=ss_type
        )
        result["feature"] = feature
        results.append(result)

    results_df = pd.DataFrame(results)

    print("[2/4] Applying FDR correction (Benjamini-Hochberg)...")

    valid_pvals = results_df["factor_p"].dropna()
    if len(valid_pvals) > 0:
        valid_indices = results_df["factor_p"].notna()
        _, adj_pvals, _, _ = multipletests(
            results_df.loc[valid_indices, "factor_p"],
            alpha=alpha,
            method="fdr_bh"
        )
        results_df.loc[valid_indices, "adj_p"] = adj_pvals
    else:
        results_df["adj_p"] = np.nan

    results_df["significant"] = results_df["adj_p"] < alpha

    results_df = results_df[[
        "feature", "n", "factor_f", "factor_p", "adj_p",
        "factor_eta_sq", "r_squared", "residual_std", "significant", "error"
    ]]

    results_df.to_csv(os.path.join(output_folder, "ancova_results.txt"), sep="\t", index=False)

    sig_count = results_df["significant"].sum()
    print(f"    Found {sig_count} significant features (FDR < {alpha})")

    print("[3/4] Generating plots...")

    generate_volcano_plot(results_df, output_folder)
    generate_effect_size_plot(results_df, output_folder)
    generate_pvalue_histogram(results_df, output_folder)

    print("[4/4] Computing adjusted means for significant features...")

    sig_features = results_df[results_df["significant"] == True]["feature"].tolist()

    if len(sig_features) > 0:
        adjusted_means_all = []
        for feature in sig_features[:100]:
            feature_data = data_long[data_long["feature"] == feature][[sample_col_name, "value"]]
            feature_data = feature_data.merge(annotation_subset, on=sample_col_name, how="inner")

            adj_means = compute_adjusted_means(
                data=feature_data,
                dependent_var="value",
                factor=factor_col,
                covariates=covariate_cols
            )
            if len(adj_means) > 0:
                adj_means["feature"] = feature
                adjusted_means_all.append(adj_means)

        if adjusted_means_all:
            adjusted_means_df = pd.concat(adjusted_means_all, ignore_index=True)
            adjusted_means_df.to_csv(
                os.path.join(output_folder, "adjusted_means.txt"),
                sep="\t", index=False
            )

    summary = {
        "total_features": len(feature_names),
        "tested_features": int(results_df["factor_p"].notna().sum()),
        "significant_features": int(sig_count),
        "fdr_threshold": alpha,
        "ss_type": ss_type,
        "factor": factor_col,
        "covariates": ",".join(covariate_cols)
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(output_folder, "analysis_summary.txt"), sep="\t", index=False)

    print("ANCOVA analysis completed successfully")
    print(f"Results saved to: {output_folder}")


@click.command()
@click.option("--input_file", "-i", required=True, help="Path to input data file")
@click.option("--annotation_file", "-a", required=True, help="Path to annotation file (Sample, Condition, covariates)")
@click.option("--output_folder", "-o", required=True, help="Path to output folder")
@click.option("--sample_cols", "-s", required=True, help="Comma-separated sample column names")
@click.option("--factor_col", "-f", required=True, help="Name of the main factor column in annotation")
@click.option("--covariate_cols", "-c", required=True, help="Comma-separated covariate column names")
@click.option("--index_col", "-x", default=None, help="Name of index column for feature IDs")
@click.option("--ss_type", "-t", type=int, default=2, help="Type of sum of squares (1, 2, or 3)")
@click.option("--alpha", type=float, default=0.05, help="FDR significance threshold")
@click.option("--log2", "-l", is_flag=True, help="Log2 transform the data")
def main(
    input_file: str,
    annotation_file: str,
    output_folder: str,
    sample_cols: str,
    factor_col: str,
    covariate_cols: str,
    index_col: str,
    ss_type: int,
    alpha: float,
    log2: bool
):
    """Batch ANCOVA analysis with multiple covariates and FDR correction."""
    sample_list = [s.strip() for s in sample_cols.split(",")]
    covariate_list = [c.strip() for c in covariate_cols.split(",") if c.strip()]

    ancova_batch(
        input_file=input_file,
        annotation_file=annotation_file,
        output_folder=output_folder,
        sample_cols=sample_list,
        factor_col=factor_col,
        covariate_cols=covariate_list,
        index_col=index_col,
        ss_type=ss_type,
        alpha=alpha,
        log2_transform=log2
    )


if __name__ == "__main__":
    main()
