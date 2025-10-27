# Donations Analytics Pipeline — Operations Guide
This handbook explains how to install, configure, and operate the donations analytics pipeline, how to automate runs on Windows, and how to interpret every worksheet in the delivered Excel workbook.
 
## 1. Overview
The pipeline performs the following end-to-end steps:
1. Load donation transactions from either a SQL Server source or a pre-extracted CSV.
2. Validate column headers and clean the dataset (duplicate removal, null handling, type coercion).
3. Build descriptive statistics (year/month trends, demographic distributions, relationship analyses).
4. Train predictive models for re-donation, major-gift propensity, gift amount, time-to-next gift, and likely fund/appeal.
5. Generate actionable donor lists (lapsed, high-potential sleepers, upgrade candidates, and new donors).
6. Package all outputs—CSV, JSON metrics, and charts—into a single Excel workbook and archive run artefacts.
 
## 2. Environment setup
### 2.1 Python and packages
1. Install **Python 3.10+**.
2. Install required packages with the provided file:
   ```bash
   pip install -r requirements.txt
   ```
   The default `requirements.txt` includes:
   - `pandas`, `numpy` for data preparation
   - `scikit-learn`, `lightgbm`, `shap` for modelling and interpretation
   - `pyodbc` for SQL Server extraction
   - `openpyxl`, `matplotlib` for Excel export and charting
 
### 2.2 External dependencies
- **SQL Server connectivity** (optional): install the Microsoft ODBC Driver 17 or 18 for SQL Server. On Windows, this is available through the official MSI installer; on Linux, install `unixODBC` plus the Microsoft driver package.

### 2.3 Environment variables and secrets
Store any sensitive values in a `.env` file located alongside `main.py`. The loader automatically reads keys such as `SQLSERVER_USER`, `SQLSERVER_PASSWORD`, `SMTP_USERNAME`, and `SMTP_PASSWORD`.

## 3. Running the pipeline
### 3.1 Command-line (CSV extract)
```bash
python -m donations_pipeline.main ^
    --input_file /path/to/FullDataTable.csv ^
    --out_dir /path/to/outputs ^
    --year_cutoff 2010 ^
    --window_days 365 ^
    --major_threshold 1000
 ```
### 3.2 Command-line (SQL Server export)
 ```bash
python -m donations_pipeline.main ^
    --use_sql true ^
    --sql_server "SPKSDB02" ^
    --sql_db "UniSA_DonorAnalysis" ^
    --window_days 365 ^
    --major_threshold 1000 ^
    --year_cutoff 2000 ^
    --out_dir "C:\Users\sysuni01\Documents\Python\Data\Output\IDR_v1" ^
    --input_file "C:\Users\sysuni01\Documents\Python\Data\Input\FullDataTable.csv" ^   # destination CSV created by the loader
    --excel_name "Donations_DataModellingResult.xlsx"
 ```

The script streams results in batches, writes the CSV named in `--input_file`, and continues with the standard analytics flow.
A ready-to-edit batch file is provided at the project root (`run_donations_pipeline.bat`). To schedule or manually trigger the pipeline on Windows:
1. Open the BAT file in a text editor.
2. Update the paths at the top for `PROJECT_ROOT`, `INPUT_FILE`, `OUTPUT_DIR`, and `VENV_DIR`.
3. Uncomment the SQL block if you prefer a direct database extract, then fill in server, database, credential, table, and driver names.
4. Double-click the BAT file or schedule it via Windows Task Scheduler. The script activates the virtual environment, runs the pipeline, prints the output folder, and pauses so that any errors remain visible.

## 4. Excel delivery workbook
The workbook `<out_dir>/since_<YEAR>/<excel_name>` consolidates every CSV and JSON artefact. A `Contents` sheet links each source file to the relevant tab, and flattened JSON metrics (e.g., model scores) appear in `json_metrics` when available.
 
### 4.1 Prospecting & predictive insights
- **`prospecting_playbook`** (from `prospecting_playbook.csv`)
- `ID`: Donor identifier.
- `pred_amount`: Expected value of the next gift from the RandomForest regression model.
- `pred_days_to_next`: Predicted days until the next donation from the time-to-next gradient boosting regressor.
- `pred_date_point`, `pred_date_p20`, `pred_date_p50`, `pred_date_p80`: Projected next-gift dates at point estimate and 20th/50th/80th percentile confidence intervals.
- `top1_fund`/`top2_fund`/`top3_fund` or `top1_appeal`…: Highest-probability destinations from the preference classifier.
- `top1_prob`/`top2_prob`/`top3_prob`: Probabilities associated with each recommended fund/appeal.
- `priority_score`: Composite rank (40% amount, 40% timing, 20% preference certainty) used to sort the playbook.
- `action_bucket`: Operational call-to-action derived from amount/timing thresholds (`VIP_target`, `Upgrade_push`, `Reactivation`, `Nurture`).
- **`propensity_model_comparison`** and **`major_gift_model_comparison`**
- Columns `model`, `auc`, `precision`, `recall`, `f1`, `support` summarise hold-out performance for Logistic Regression, Random Forest, and LightGBM classifiers.
- Higher `auc` or `f1` indicates a stronger candidate for scoring.
- **`field_importance_propensity`**, **`field_importance_major_gift`**, etc.
- Each row shows a source `field` with aggregated importance across one-hot encoded levels.
- Importances are normalised to sum to 1.0 for the underlying tree-based model; higher values indicate stronger predictive power.
- Use these sheets to explain which donor attributes drive the scores. Fields absent from the sheet either lacked predictive signal or were unavailable in the training slice.

### 4.2 RFM segmentation & lapsed donors
- **`rfm_table`**
- `Recency`: Days since last gift (0 indicates same-day gifts).
- `Frequency`: Total count of recorded gifts.
- `Monetary`: Lifetime donation amount.
- `LastAmount`, `FirstGiftDate`, `LastGiftDate`: Sanity-check columns for donor histories.
- `R_score`, `F_score`, `M_score`, `RFM_Score`, `RFM_Segment`: Quintile-based scores and canonical segment labels (Champion, Loyal, At Risk, etc.).
- **`lapsed_targets`**
- Contains donors with `Recency ≥ 365` and `Monetary ≥ 100`.
- Shares the same numeric columns as `rfm_table`, allowing quick filtering or mail-merge exports.

### 4.3 Actionable donor lists
All actionable tabs originate from `actionable_lists.py` and share the following core columns:
| Column                     | Meaning                                                                                                                        |
| -------------------------  | ------------------------------------------------------------------------------------------------------------------------------ |
| `ID`                       | Donor identifier.                                                                                                              |
| `Recency`                  | Days since the most recent gift at the time of the run.                                                                        |
| `Frequency`                | Total number of gifts on record.                                                                                               |
| `Monetary`                 | Lifetime donation value.                                                                                                       |
| `AvgAmount`                | Arithmetic mean of all gifts.                                                                                                  |
| `LastAmount`               | Amount of the most recent gift.                                                                                                |
| `LastGiftDate`             | Date of the most recent gift.                                                                                                  |
| `FirstGiftDate`            | Donor’s first recorded gift date.                                                                                              |
| `Gifts12`, `Amt12`         | Number of gifts and total amount in the most recent 365-day window.                                                            |
| `GiftsPrev12`, `AmtPrev12` | Number of gifts and total amount in the preceding 365-day window (days 366–730 before the reference date). Values of 0 indicate no activity in that period. |

Specific logic per tab:
- `lapsed_12m`: Donors inactive for ≥12 months but historically valuable (`Monetary ≥ 100`).
- `high_potential_sleeper`: Donors inactive for ≥12 months with stronger history (`Monetary ≥ 500`). Use `Gifts12` vs `GiftsPrev12` to spot declining engagement.
- `upgrade_candidates`: Donors whose last gift is at least 1.5× their average gift and who made ≥2 gifts in total. Prioritise those with high `AvgAmount` and recent `LastGiftDate`.
- `new_donors_90d`: Donors whose first gift landed within the configured `new_donor_days` (default 90). Columns `GiftsPrev12`/`AmtPrev12` are absent by design.
 
### 4.4 Descriptive trend & distribution tabs
- `trend_year` / `trend_month`: Time-series aggregates for amount, donors, and counts, including year-over-year growth metrics.
- `dist_gender`, `dist_agegroup`, `dist_occupation`, `dist_religion`, `dist_suburb`: Category-level totals, donor counts, and averages.
- `appeal_performance`, `fund_performance`: Campaign/fund breakdowns using the same measures as the demographic tabs.

### 4.5 Relationship analytics
- `relationship_summary`: Donation totals by relationship flag combination (e.g., Current Parent, Past Student).
- `relationship_trend`: Yearly totals for each relationship combination, allowing you to track engagement shifts.
 
### 4.6 Model metrics (`json_metrics` sheet)
Metric JSON files saved during training are flattened here. Expect regression metrics such as `MAE`, `RMSE`, `R2`, along with classification metrics (`auc`, `precision`, `recall`). Use these values to compare runs over time.
 
## 5. Understanding the predictive models
The modelling pipeline trains multiple algorithms and automatically exports metrics and field importances:
- **Re-donation propensity**: Logistic Regression (baseline), RandomForestClassifier, and LightGBMClassifier. Scores represent the probability of a donor giving again within the analysis window. Focus on donors with high scores but high `Recency` for reactivation opportunities.
- **Major gift propensity**: Same algorithm roster as above, but with the target defined by `--major_threshold`. Use the resulting probabilities to build cultivation plans for major gift officers.
- **Next gift amount**: RandomForestRegressor predicts the expected value of the next donation (`pred_amount`). Compare against `LastAmount` to find potential upgrades.
- **Time to next gift**: Gradient boosting regressor estimates days until the next gift (`pred_days_to_next`). Negative or very small values indicate imminent gifts; large values identify lapsed risk.
- **Preference model**: When `Fund` or `Appeal` data is dense, a multi-class classifier ranks likely destinations. Low confidence across all classes suggests messaging should emphasise general appeals.
 
Feature importances are aggregated to the original field name. A field with importance 0.25 means roughly 25% of the splitting power in the tree-based model came from that column (summing to 1.0). Use this to explain why the model prioritises certain segments (e.g., `Recency`, `AvgAmount`, or relationship flags).

## 6. Post-run housekeeping & troubleshooting
- The Excel bundle auto-increments (`_v2`, `_v3`, …) to avoid overwriting earlier deliverables.
- Source CSV, JSON, and model artefacts move into a dated `archive_YYYYMMDD` folder for traceability.
- Use `archive_YYYYMMDD/models/` to retrieve the pickled scikit-learn pipelines if you need to score donors externally.
 
Common issues:
| `pyodbc` import error | ODBC driver missing | Install Microsoft ODBC Driver 17/18 for SQL Server and re-run `pip install -r requirements.txt`. |
| Empty predictive sheets | Training data too small or no positive labels | Check `pipeline.log` for warnings, widen `--year_cutoff`, or ensure major-gift threshold matches your data. |
| Missing demographic/relationship tabs | Source columns not present | Confirm the source extract includes the demographic fields before running. |

With this guide you can install the environment, automate execution, and deliver the workbook with enough context for stakeholders to interpret every sheet and model output.
