# ANCOVA Analysis


## Installation

**[⬇️ Click here to install in Cauldron](http://localhost:50060/install?repo=https%3A%2F%2Fgithub.com%2Fnoatgnu%2Fancova-plugin)** _(requires Cauldron to be running)_

> **Repository**: `https://github.com/noatgnu/ancova-plugin`

**Manual installation:**

1. Open Cauldron
2. Go to **Plugins** → **Install from Repository**
3. Paste: `https://github.com/noatgnu/ancova-plugin`
4. Click **Install**

**ID**: `ancova-analysis`  
**Version**: 1.0.0  
**Category**: analysis  
**Author**: CauldronGO Team

## Description

Analysis of Covariance (ANCOVA) for comparing group means while controlling for covariates, with FDR correction


## Workflow Diagram

```mermaid
flowchart TD
    Start([Start]) --> step1
    step1[Running ANCOVA for {len(feature_names]
    step1 --> step2
    step2[Applying FDR correction (Benjamini-Hochberg]
    step2 --> step3
    step3[Generating plots...]
    step3 --> step4
    step4[Computing adjusted means for significant features...]
    step4 --> End([End])
```

## Runtime

- **Environments**: `python`

- **Entrypoint**: `ancova_analysis.py`

## Inputs

| Name | Label | Type | Required | Default | Visibility |
|------|-------|------|----------|---------|------------|
| `input_file` | Input Data File | file | Yes | - | Always visible |
| `annotation_file` | Annotation File | file | Yes | - | Always visible |
| `sample_cols` | Sample Columns | column-selector (multiple) | Yes | - | Always visible |
| `index_col` | Feature ID Column | column-selector (single) | No | - | Always visible |
| `factor_col` | Main Factor Column | column-selector (single) | Yes | - | Always visible |
| `covariate_cols` | Covariate Columns | column-selector (multiple) | Yes | - | Always visible |
| `ss_type` | Sum of Squares Type | select (Type I (Sequential), Type II (Hierarchical), Type III (Marginal)) | No | 2 | Always visible |
| `alpha` | FDR Threshold | number (min: 0, max: 0, step: 0) | No | 0.05 | Always visible |
| `log2` | Log2 Transform | boolean | No | false | Always visible |

### Input Details

#### Input Data File (`input_file`)

Data matrix with features as rows and samples as columns


#### Annotation File (`annotation_file`)

Sample annotation file with Sample, Condition, and covariate columns


#### Sample Columns (`sample_cols`)

Select columns containing sample data

- **Column Source**: `input_file`

#### Feature ID Column (`index_col`)

Column containing feature identifiers (e.g., protein IDs)

- **Column Source**: `input_file`

#### Main Factor Column (`factor_col`)

Column in annotation file containing the main grouping factor (e.g., Condition)

- **Column Source**: `annotation_file`

#### Covariate Columns (`covariate_cols`)

Columns in annotation file containing covariates to control for (e.g., Batch, Age)

- **Column Source**: `annotation_file`

#### Sum of Squares Type (`ss_type`)

Type of sum of squares for ANCOVA

- **Options**: `1` (Type I (Sequential)), `2` (Type II (Hierarchical)), `3` (Type III (Marginal))

#### FDR Threshold (`alpha`)

False Discovery Rate threshold (Benjamini-Hochberg)


#### Log2 Transform (`log2`)

Apply log2 transformation to data before analysis


## Outputs

| Name | File | Type | Format | Description |
|------|------|------|--------|-------------|
| `ancova_results` | `ancova_results.txt` | data | tsv | ANCOVA results with F-statistics, p-values, adjusted p-values, and effect sizes |
| `adjusted_means` | `adjusted_means.txt` | data | tsv | Covariate-adjusted means for significant features |
| `analysis_summary` | `analysis_summary.txt` | data | tsv | Summary of analysis parameters and results |
| `volcano_plot` | `volcano_plot.html` | html | html | Volcano plot showing effect size vs significance |
| `effect_size_plot` | `effect_size_plot.html` | html | html | Bar plot of top features by effect size |
| `pvalue_histogram` | `pvalue_histogram.html` | html | html | Distribution of raw and adjusted p-values |

## Sample Annotation

This plugin supports sample annotation:

- **Samples From**: `sample_cols`
- **Annotation File**: `annotation_file`

## Visualizations

This plugin generates 1 plot(s):

### Volcano Plot (`ancova-volcano`)

- **Type**: scatter
- **Data Source**: `ancova_results`
- **Default**: Yes

## Requirements

- **Python Version**: >=3.11

### Package Dependencies (Inline)

Packages are defined inline in the plugin configuration:

- `numpy>=1.24.0`
- `pandas>=2.0.0`
- `statsmodels>=0.14.0`
- `plotly>=5.18.0`
- `click>=8.0.0`

> **Note**: When you create a custom environment for this plugin, these dependencies will be automatically installed.

## Example Data

This plugin includes example data for testing:

```yaml
  annotation_file: differential_analysis/batch_info.txt
  sample_cols: [C:\Raja\DIA-NN searches\June 2022\LT-CBQCA-Test_DIA\RN-DS_220106_BCA_LT-IP_01.raw C:\Raja\DIA-NN searches\June 2022\LT-CBQCA-Test_DIA\RN-DS_220106_BCA_LT-IP_02.raw C:\Raja\DIA-NN searches\June 2022\LT-CBQCA-Test_DIA\RN-DS_220106_BCA_LT-IP_03.raw C:\Raja\DIA-NN searches\June 2022\LT-CBQCA-Test_DIA\RN-DS_220106_BCA_LT-MockIP_01.raw C:\Raja\DIA-NN searches\June 2022\LT-CBQCA-Test_DIA\RN-DS_220106_BCA_LT-MockIP_02.raw C:\Raja\DIA-NN searches\June 2022\LT-CBQCA-Test_DIA\RN-DS_220106_BCA_LT-MockIP_03.raw]
  factor_col: Condition
  covariate_cols: [Batch]
  alpha: 0.05
  log2: true
  input_file: diann/imputed.data.txt
  sample_cols_source: diann/imputed.data.txt
  ss_type: 2
```

Load example data by clicking the **Load Example** button in the UI.

## Usage

### Via UI

1. Navigate to **analysis** → **ANCOVA Analysis**
2. Fill in the required inputs
3. Click **Run Analysis**

### Via Plugin System

```typescript
const jobId = await pluginService.executePlugin('ancova-analysis', {
  // Add parameters here
});
```
