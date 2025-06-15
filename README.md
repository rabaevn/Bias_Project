# Fairness Evaluation via Counterfactual Perturbations

This repository contains a modular, reproducible pipeline for evaluating machine learning model fairness on the UCI Adult dataset using counterfactual perturbations to sensitive attributes (sex and race). The project was designed for academic research and paper submission.

## 📁 Project Structure

```
.
├── src/
│   ├── preprocess.py          # Preprocesses and encodes the dataset
│   ├── baseline_models.py     # Trains classifiers and evaluates fairness
│   ├── counterfactual.py      # Perturbs protected attributes and analyzes fairness
│   └── ttest_analysis.py      # Performs t-tests to assess statistical significance
├── data/
│   └── adult.csv              # UCI Adult dataset (not included — download separately)
├── requirements.txt           # Python package dependencies
└── README.md
```

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # on Windows use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Usage Instructions

### 1. Preprocess the data
```python
from src.preprocess import load_and_preprocess_data
X_train, y_train, X_test, y_test, prot_attr_train, prot_attr_test = load_and_preprocess_data()
```

### 2. Train baseline models and evaluate fairness
```python
from src.baseline_models import evaluate_models
from aif360.datasets import StandardDataset
results_df = evaluate_models(StandardDataset)  # Use an AIF360-formatted dataset
```

### 3. Run counterfactual experiments
```python
from src.counterfactual import run_counterfactual_experiment
import pandas as pd

df = pd.read_csv("data/adult.csv")
run_counterfactual_experiment(df, protected_attr='sex', output_prefix='gender')
run_counterfactual_experiment(df, protected_attr='race', output_prefix='race')
```

### 4. Conduct t-tests on model performance
```python
from src.ttest_analysis import run_ttest_on_predictions
df_preds = pd.read_csv("counterfactual_success_log.csv")
run_ttest_on_predictions(df_preds)
```

## 🧾 Dataset

Download the [UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult) and save it as `data/adult.csv`.

## 📄 License

This code is provided for academic and research use. Please cite or acknowledge if used in published work.

---