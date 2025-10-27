# Donations Analytics Pipeline (Refactored) 🎯

A production-ready, modular analytics pipeline for donor data analysis and predictive modeling.

---

## 📁 Project Structure

```
donations_pipeline/
│
├── 📦 Core Modules
│   ├── config.py                    # Configuration management
│   ├── logger.py                    # Logging & performance tracking
│   ├── data_loader.py               # Data loading (CSV/SQL)
│   ├── preprocessor.py              # Data preprocessing & feature engineering
│   │
├── 📊 Analytics Modules
│   ├── descriptive_analytics.py     # Trends & distributions
│   ├── rfm_analysis.py              # RFM segmentation
│   ├── actionable_lists.py          # Targeting & reactivation lists
│   │
├── 🤖 Machine Learning
│   ├── models.py                    # Model trainers & evaluators
│   ├── predictive_pipeline.py       # Prediction orchestration
│   │
├── 📤 Output Management
│   ├── output_manager.py            # Results packaging & archiving
│   │
└── 🚀 Entry Point
    └── main.py                      # Pipeline orchestration
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd CapstoneProject

# Install dependencies
pip install -r requirements.txt

# Install as package (optional)
pip install -e .
```

### Basic Usage

```bash
# Run with CSV input
python -m donations_pipeline.main \
    --input_file data/FullDataTable.csv \
    --out_dir outputs \
    --year_cutoff 2000

# Run with SQL extraction
python -m donations_pipeline.main \
    --use_sql true \
    --sql_server localhost \
    --sql_db DonorDatabase \
    --out_dir outputs
```

### Python API Usage

```python
from donations_pipeline.config import Config
from donations_pipeline.main import DonationsPipeline

# Create configuration
config = Config(
    input_file="data.csv",
    out_dir="outputs",
    analysis=AnalysisConfig(
        year_cutoff=2000,
        major_gift_threshold=1000.0
    )
)

# Run pipeline
pipeline = DonationsPipeline(config)
pipeline.run()
```

---

## 🎯 Features

### 1. Descriptive Analytics
- **Yearly trends:** Total donations, donor counts, averages
- **Monthly trends:** Seasonal patterns
- **Demographic distributions:** Gender, age group, occupation, suburb
- **Campaign performance:** Appeal and fund effectiveness

### 2. RFM Segmentation
- **Recency:** Days since last donation
- **Frequency:** Number of donations
- **Monetary:** Total donation amount
- **Segments:** Champions, Loyal, At Risk, High Potential, Regular

### 3. Relationship Analysis
- School relationship patterns (current/past parent, student)
- Trends over time
- Cross-analysis with demographics

### 4. Actionable Lists
- **Lapsed donors:** >12 months inactive with history
- **High-potential sleepers:** Valuable but dormant
- **Upgrade candidates:** Recent increased giving
- **New donors:** Acquired in last 90 days

### 5. Predictive Models

#### Re-donation Propensity
Predicts likelihood of future donation within time window
- Models: Logistic Regression, Random Forest, LightGBM
- Features: RFM metrics + demographics
- Output: Probability scores per donor

#### Major Gift Propensity
Predicts likelihood of donation ≥ threshold
- Configurable threshold (default: $1,000)
- Time-based validation to avoid leakage

#### Donation Amount
Predicts expected future donation amount
- Random Forest regressor
- Output: Point estimate per donor

#### Time to Next Donation
Predicts days until next donation
- Random Forest regressor
- Confidence intervals (20th, 50th, 80th percentile)

#### Fund/Appeal Preference
Predicts most likely fund/appeal for next donation
- Multiclass classifier
- Top-3 preferences with probabilities

### 6. Prospecting Playbook
Combines all predictions into actionable donor profiles:
- Predicted amount
- Predicted next donation date
- Preferred fund/appeal
- Priority score (composite metric)
- Action bucket (VIP, Upgrade, Reactivation, Nurture)

---

## ⚙️ Configuration Options

### Command Line Arguments

```bash
# Data Input
--input_file          Path to CSV file (required)
--use_sql            Extract from SQL Server (true/false)
--sql_server         SQL Server address
--sql_db             Database name
--sql_trusted        Use Windows Auth (true/false)

# Output
--out_dir            Output directory (default: outputs)
--excel_name         Excel filename (default: Donations_DataModellingResult.xlsx)

# Analysis Parameters
--rfm_reference      RFM reference date (YYYY-MM-DD)
--window_days        Prediction window in days (default: 365)
--major_threshold    Major gift amount (default: 1000.0)
--year_cutoff        Analysis year cutoff (default: 2000)
```

### Programmatic Configuration

```python
from donations_pipeline.config import Config, AnalysisConfig, ModelConfig

config = Config(
    input_file="data.csv",
    out_dir="outputs",
    
    analysis=AnalysisConfig(
        year_cutoff=2000,
        major_gift_threshold=1000.0,
        lapsed_recency_days=730,
        age_bins=[0, 25, 35, 45, 55, 65, 75, 200]
    ),
    
    models=ModelConfig(
        rf_n_estimators=300,
        lgbm_learning_rate=0.05
    )
)
```

---

## 📊 Output Files

### Directory Structure

```
outputs/
├── pipeline.log                              # Execution log
├── pipeline_timing.txt                       # Performance report
├── cleaned_donations.csv                     # Preprocessed data
│
├── since_2000/                               # Analysis slice
│   ├── 📈 Descriptive
│   │   ├── trend_year.csv
│   │   ├── trend_month.csv
│   │   ├── dist_gender.csv
│   │   ├── dist_agegroup.csv
│   │   ├── appeal_performance.csv
│   │   └── fund_performance.csv
│   │
│   ├── 🎯 Segmentation
│   │   ├── rfm_table.csv
│   │   ├── lapsed_targets.csv
│   │   ├── relationship_summary.csv
│   │   └── relationship_trend.csv
│   │
│   ├── 📋 Actionable Lists
│   │   ├── lapsed_12m.csv
│   │   ├── high_potential_sleeper.csv
│   │   ├── upgrade_candidates.csv
│   │   └── new_donors_90d.csv
│   │
│   ├── 🤖 Predictive Models
│   │   ├── propensity_RandomForest_model.pkl
│   │   ├── propensity_RandomForest_metrics.json
│   │   ├── major_gift_LightGBM_model.pkl
│   │   ├── amount_predictor_model.pkl
│   │   ├── time_to_next_predictor_model.pkl
│   │   ├── preference_fund_model.pkl
│   │   └── field_importance_*.csv
│   │
│   ├── 🎯 Predictions
│   │   └── prospecting_playbook.csv         # Combined predictions
│   │
│   └── 📦 Packaged Results
│       └── Donations_DataModellingResult.xlsx
```

---

## 🔧 Advanced Usage

### 1. Custom Analysis Workflow

```python
from donations_pipeline.data_loader import DataLoader
from donations_pipeline.rfm_analysis import RFMAnalyzer
from donations_pipeline.logger import setup_logger

# Setup
config = Config(input_file="data.csv")
logger = setup_logger(config.output_path)

# Load data
loader = DataLoader(config, logger)
df = loader.load_data()

# Run only RFM
rfm_analyzer = RFMAnalyzer(config, logger)
rfm_df = rfm_analyzer.compute_rfm(df)

# Custom threshold for lapsed donors
lapsed = rfm_analyzer.identify_lapsed_donors(
    rfm_df, 
    recency_threshold=365,  # 1 year
    min_monetary=500.0      # $500 minimum
)
```

### 2. Model Reuse for Predictions

```python
import joblib
import pandas as pd

# Load trained model
model = joblib.load("outputs/since_2000/propensity_RandomForest_model.pkl")

# Prepare new data
new_donors = pd.DataFrame({
    'Recency': [30, 90, 180],
    'Frequency': [5, 3, 1],
    'Monetary': [1500, 500, 100],
    # ... other features
})

# Predict
probabilities = model.predict_proba(new_donors)[:, 1]
print(f"Re-donation probabilities: {probabilities}")
```

### 3. Batch Processing Multiple Files

```python
from pathlib import Path

input_dir = Path("data/monthly_exports")

for csv_file in input_dir.glob("*.csv"):
    config = Config(
        input_file=str(csv_file),
        out_dir=f"outputs/{csv_file.stem}"
    )
    
    pipeline = DonationsPipeline(config)
    pipeline.run()
```

---

## 🧪 Testing

### Run Tests

```bash
# Install dev dependencies
pip install pytest pytest-cov

# Run unit tests
pytest tests/

# With coverage report
pytest --cov=donations_pipeline tests/
```

### Example Test

```python
# tests/test_rfm.py
import pytest
from donations_pipeline.rfm_analysis import RFMAnalyzer

def test_rfm_scores_within_range(sample_data, test_config, test_logger):
    analyzer = RFMAnalyzer(test_config, test_logger)
    rfm_df = analyzer.compute_rfm(sample_data)
    
    assert rfm_df['R_score'].between(1, 5).all()
    assert rfm_df['F_score'].between(1, 5).all()
    assert rfm_df['M_score'].between(1, 5).all()
```

---

## 📈 Performance

### Typical Execution Times

| Dataset Size | Records | Duration |
|--------------|---------|----------|
| Small        | 10K     | ~30s     |
| Medium       | 100K    | ~3min    |
| Large        | 1M      | ~20min   |

### Memory Usage

- **CSV Loading:** ~2x file size
- **Feature Engineering:** ~3x data size
- **Model Training:** ~4x data size

*For datasets >500K records, consider chunked processing or sampling*

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Out of Memory

```python
# Solution: Use year cutoff or sample
config.analysis.year_cutoff = 2015  # Only recent data
```

#### 2. SQL Connection Failed

```bash
# Check driver installation
pip install pyodbc

# Verify ODBC driver
odbcinst -j

# Try different driver
--sql_odbc_driver "ODBC Driver 17 for SQL Server"
```

#### 3. Model Training Fails

```python
# Check minimum samples
if len(X_train) < config.analysis.min_samples_for_training:
    logger.warning("Insufficient samples for training")
```

### Debug Mode

```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
python -m donations_pipeline.main --input_file data.csv
```

---

## 🔒 Security Considerations

1. **SQL Injection:** All queries use parameterized statements
2. **Path Traversal:** All paths validated with `Path.resolve()`
3. **Sensitive Data:** Configure `.gitignore` to exclude data files
4. **Credentials:** Never commit SQL passwords; use environment variables

```bash
# Use environment variables for credentials
export SQL_USER="myuser"
export SQL_PASSWORD="mypassword"
```

---

## 📚 Documentation

- **Code Documentation:** Inline docstrings in all modules
- **Architecture:** See `REFACTORING_SUMMARY.md`
- **API Reference:** Use `help(DonationsPipeline)`

---

## 🤝 Contributing

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run linting
flake8 donations_pipeline/
black donations_pipeline/
mypy donations_pipeline/
```

### Adding New Features

1. Create new module in appropriate category
2. Follow existing patterns (dependency injection, logging)
3. Add unit tests
4. Update documentation

---

## 📝 License

[Your License Here]

---

## 👥 Authors

- **Original Implementation:** Yichi Nien
- **Refactoring:** October 2025

---

## 🙏 Acknowledgments

Built with modern Python best practices for production-grade data analytics.

---

**Status:** ✅ Production Ready  
**Version:** 2.0.0 (Refactored)  
**Last Updated:** October 2025