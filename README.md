# Medical Insurance Cost Prediction

A Linear Regression model that estimates a customer's annual medical insurance charges from demographic and health attributes, built as part of my ML internship at Big Brains Learning.

**Live app:** (https://hassan-medical-insurance-cost-prediction.streamlit.app/)

## Problem

Health insurers price policies based on expected future claims. Charging every customer the same premium is unsustainable — high-risk customers cost more than they pay in, and low-risk customers get overcharged and leave for competitors. This project estimates expected annual medical charges for a customer at enrollment, using information available upfront (no claims history needed).

This is a **regression** problem: the target, `charges`, is a continuous dollar value.

**Note on real-world use:** insurers use models like this for risk-based pricing and underwriting, but in practice they're legally restricted in which features they can price on. A production version of this model would need to account for those regulatory constraints — this project is a technical exercise, not a deployable pricing engine.

## Dataset

- **Source:** [Kaggle — Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance) (mirichoi0218)
- **Size:** 1,338 rows × 7 columns
- **Features:** `age`, `sex`, `bmi`, `children`, `smoker`, `region`
- **Target:** `charges`

## Preprocessing

- Checked for missing values — none found (all 7 columns fully populated)
- Found and removed **1 duplicate row** (1338 → 1337 rows)
- Encoded categoricals for modeling:
  - `sex` and `smoker` mapped to binary (0/1) — only two categories each, so no need for one-hot encoding
  - `region` one-hot encoded with `drop_first=True` (4 categories → 3 columns; `northeast` is the implicit baseline, avoiding the dummy-variable trap)

## Exploratory Data Analysis

Key findings from correlation analysis and visualizations:

| Feature | Correlation with `charges` |
|---|---|
| smoker | **0.79** |
| age | 0.30 |
| bmi | 0.20 |
| children | 0.07 |
| sex | 0.06 |

- **Smoking status is the dominant driver of charges** — smokers' median charges (~$34k) are roughly 3–4x non-smokers' (~$9–10k), with almost no overlap between the two groups.
- **BMI's effect is conditional, not linear on its own.** For non-smokers, BMI barely moves charges. For smokers, charges rise sharply once BMI crosses ~30 (the obesity threshold) — a real interaction effect that a plain correlation coefficient understates.
- **Age has a moderate, mostly linear relationship**, but the age-vs-charges scatter shows three distinct diagonal bands rather than one cloud — driven by the same smoker/BMI interaction.
- `sex`, `children`, and `region` show weak-to-negligible correlation with charges individually.

## Model

- **Algorithm:** Multiple Linear Regression (scikit-learn)
- **Features:** 8 columns after encoding (`age`, `sex`, `bmi`, `children`, `smoker`, `region_northwest`, `region_southeast`, `region_southwest`)
- **Train/test split:** 80/20, `random_state=42`
- **Smoker coefficient:** ~$23,078 — being a smoker adds roughly this much to predicted charges holding all else constant, consistent with the 0.79 correlation found in EDA

## Evaluation

| Metric | Value |
|---|---|
| MAE | $4,177.05 |
| MSE | 35,478,020.68 |
| RMSE | $5,956.34 |
| R² Score | 0.8069 |

**Interpretation:** R² of 0.81 means the model explains about 81% of the variance in charges — solid for a plain linear regression on this dataset. The gap between RMSE and MAE (RMSE is ~1.4x MAE) shows a minority of predictions carry a disproportionate share of the error, which lines up with the BMI×smoker interaction identified in EDA: the model systematically over- or under-predicts for customers where that interaction is strongest (e.g. high-BMI non-smokers, or low-BMI smokers).

## Limitations

- As a **linear** model, it cannot natively capture the BMI×smoker interaction found in EDA — an interaction term (`bmi * smoker`) or a non-linear model (e.g. Random Forest, Gradient Boosting) would likely reduce error further.
- Trained on a single, relatively small (1,337-row) public dataset — not validated against real-world claims data, and regional/demographic patterns may not generalize beyond the dataset's source population.
- Does not account for pre-existing conditions, claims history, or other underwriting factors real insurers use.

## Prediction App

A Streamlit app (`app/predict_app.py`) takes age, sex, BMI, children, smoker status, and region as input and returns an estimated charge using the trained model.

### Run locally

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cd app
streamlit run predict_app.py
```

## Project Structure

```
medical-insurance-cost-prediction/
├── data/
│   ├── insurance.csv
│   └── insurance_cleaned.csv
├── notebooks/
│   ├── 01_eda.ipynb          # Phases 1-4: problem framing, exploration, cleaning, EDA
│   └── 02_modeling.ipynb     # Phases 5-6: training, evaluation
├── app/
│   ├── predict_app.py
│   ├── model.pkl
│   └── model_columns.pkl
├── requirements.txt
└── README.md
```