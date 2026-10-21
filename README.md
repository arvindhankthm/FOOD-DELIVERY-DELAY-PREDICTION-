# Food Delivery Delay Prediction

A beginner-friendly machine learning project that predicts how many minutes a food delivery will be late, built with Python and scikit-learn.

## Overview

Late deliveries hurt customer satisfaction. This project trains a **Gradient Boosting Regressor** on order, courier and condition features to estimate delivery delay, and shows which factors affect delays the most.

> **Note:** The dataset is synthetically generated so the project runs out of the box. Results on real-world data will differ.

## Features Used

| Feature | Type | Description |
|---|---|---|
| `distance_km` | Numeric | Distance from restaurant to customer |
| `prep_time_min` | Numeric | Restaurant preparation time |
| `num_items` | Numeric | Number of items in the order |
| `courier_experience_months` | Numeric | Courier's experience |
| `hour` | Numeric | Hour of the day the order was placed |
| `traffic` | Categorical | low / medium / high |
| `weather` | Categorical | clear / rain / storm |
| `is_weekend` | Categorical | yes / no |

**Target:** `delay_min` (delay in minutes versus the promised ETA)

## How It Works

1. Generate 5,000 synthetic delivery records
2. Preprocess with a scikit-learn `Pipeline` (`StandardScaler` for numeric, `OneHotEncoder` for categorical features)
3. Train a `GradientBoostingRegressor` on an 80/20 train/test split
4. Evaluate with MAE and R²
5. Rank features using permutation importance
6. Predict the delay for a sample order

## Results

| Metric | Value |
|---|---|
| Mean Absolute Error (MAE) | ~2.13 min |
| R² Score | ~0.81 |

Most important factors: **distance, traffic and weather**.

Sample output:

```
MAE: 2.13 min
R²:  0.810

Feature importance (permutation):
distance_km                  0.471
traffic                      0.437
weather                      0.271
hour                         0.157
prep_time_min                0.139
num_items                    0.063
courier_experience_months    0.046
is_weekend                   0.038

Predicted delay for sample order: 22.2 min
```

## Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/arvindhankthm/FOOD-DELIVERY-DELAY-PREDICTION-.git
cd FOOD-DELIVERY-DELAY-PREDICTION-
```

**2. Install dependencies**

```bash
python -m pip install numpy pandas scikit-learn
```

**3. Run the model**

```bash
python delivery_delay_model.py
```

## Predict Your Own Order

Edit the `new_order` dictionary at the bottom of `delivery_delay_model.py`:

```python
new_order = pd.DataFrame([{
    "distance_km": 6.5, "prep_time_min": 22, "num_items": 5,
    "courier_experience_months": 4, "hour": 19,
    "traffic": "high", "weather": "rain", "is_weekend": "no",
}])
```

## Using Real Data

Replace `make_data()` with `pd.read_csv("your_data.csv")` and update the `NUMERIC` and `CATEGORICAL` lists to match your columns.

## Tech Stack

Python · pandas · NumPy · scikit-learn

## Future Improvements

- Train on a real delivery dataset
- Add a classifier for "late vs on-time"
- Compare models (Random Forest, XGBoost, Linear Regression)
- Tune hyperparameters with cross-validation
- Build a simple Streamlit or Flask web app

## Author

**arvindhankthm** · [GitHub](https://github.com/arvindhankthm)
