# Flight Disruption Prediction System — v2.0

*Machine Learning | Python | CatBoost | XGBoost | FastAPI | Docker*

---

### **Overview**

This project predicts **flight delays and cancellations** using machine learning models trained on U.S. domestic flight data and hourly weather data from NOAA.

Originally developed in **2024** as a notebook-based exploration using logistic regression and random forests, this **v2.0** rebuild (started in August 2025) incorporates:

* Real **meteorological data** from 362 airports via NOAA’s Integrated Surface Database (ISD)
* Improved **feature engineering** (6-hour weather window before/after flights)
* Production-ready deployment via **FastAPI** and **Docker**

---

### **Project Motivation**

The earlier version achieved high classification accuracy but relied on limited weather indicators embedded within the flight dataset.
This version focuses on integrating **true hourly weather signals** — temperature, visibility, wind, and coded meteorological events — to improve the realism and predictive value of the model.

---

### **Data Sources**

* **Flight Data (2018)**: U.S. Bureau of Transportation Statistics (BTS) — 5M+ domestic records.
* **Weather Data**: NOAA ISD hourly observations for 362 airports.
* **Airport Mapping**: IATA–NOAA lookup table to link flight origins/destinations with weather stations.

Each flight was matched with weather conditions **6 hours before and after** the scheduled departure/arrival time.

---

### **Version 1 (2024 Recap)**

* Built and tested logistic regression, SGD, Ridge, Lasso, and Random Forest models.
* Preprocessed data with imputation for missing delays and label encoding for categorical variables.
* Achieved:

  * **F1-score = 0.99** (cancellations — after tuning)
  * **R² ≈ 0.35** (delay regression)
* Explored correlation matrices, time-based feature engineering, and undersampling for class balance.

---

### **Version 2 (2025 — Current Implementation)**

#### **Feature Engineering**

* **Continuous variables:** interpolated pressure (SLP), wind speed, visibility, temperature, dew point.
* **Coded variables:** mapped event codes (e.g., fog, snow, rain) as categorical.
* **Aggregations:**

  * Mean of numeric features over the ±6h window.
  * Mode of categorical features for dominant condition.
* Combined **departure** and **arrival** weather snapshots with flight metadata (month, day, airline, distance, etc.).

#### **Modeling**

| Task                        | Model              | Metric   | Result   |
| --------------------------- | ------------------ | -------- | -------- |
| Cancellation Classification | CatBoostClassifier | F1-score | **0.84** |
| Delay Duration Regression   | XGBRegressor       | R²       | **0.41** |

**Model design:**

* CatBoost selected for native categorical handling.
* XGBoost for continuous regression.
* Parameter tuning via randomized grid search (depth, learning_rate, iterations).
* Temporal CV (month-based splits) to prevent leakage.

#### **Deployment**

* **FastAPI backend** exposes `/predict` for real-time disruption scoring.
* **Dockerized** for scalable cloud deployment.

**Example request:**

```bash
POST /predict
{
  "origin": "ATL",
  "dest": "ORD",
  "airline": "UA",
  "sched_dep_hour": 16,
  "month": 9,
  "distance": 1100,
  "temperature_origin": 29.4,
  "wind_speed_origin": 6.2,
  "visibility_origin": 8.5,
  "temperature_dest": 24.1,
  "wind_speed_dest": 5.0,
  "visibility_dest": 10.0
}
```

**Response:**

```json
{
  "cancellation_probability": 0.21,
  "expected_delay_minutes": 17.8
}
```

---

### **Results & Insights**

* Integration of true weather signals improved F1-score by **~10%** compared to the earlier dataset.
* Strongest predictors: *departure hour*, *airline*, *visibility*, *pressure variation*.
* Delay prediction remains noisy (R² ≈ 0.41) due to operational randomness — consistent with industry benchmarks.

---

### **Tech Stack**

* **Python:** pandas, numpy, scikit-learn, CatBoost, XGBoost
* **API:** FastAPI
* **Containerization:** Docker
* **Visualization:** Matplotlib, Seaborn
* **Data Storage:** CSV, feather (for intermediate steps)

---

### **How to Run**

**Local setup**

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

**Docker**

```bash
docker build -t flight-predictor .
docker run -p 8000:8000 flight-predictor
```

API documentation available at **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

### **Project Timeline**

| Phase                   | Period       | Description                                            |
| ----------------------- | ------------ | ------------------------------------------------------ |
| V1: Model Prototyping   | Jan–Jun 2024 | Logistic Regression, Random Forest, baseline analysis  |
| V2: Weather Integration | Aug–Oct 2025 | NOAA weather integration, CatBoost/XGBoost training    |
| V2.1: Deployment        | Planned      | FastAPI + Docker containerization for live predictions |

---

### **Repository Structure**

```
├── app.py                  # FastAPI app for inference
├── Dockerfile              # Container definition
├── scripts/
│   ├── clean_weather.py
│   ├── merge_flight_weather.py
├── models/
│   ├── catboost_cancel.cbm
│   ├── xgb_delay.json
├── notebooks/
│   ├── feature_engineering.ipynb
│   ├── model_training.ipynb
├── data/
│   ├── flight_data_2018.csv
│   ├── weather_cleaned/
│   ├── weather_final/
└── README.md
```

---

### **Future Enhancements**

* Add real-time NOAA API fetch for live predictions.
* Feature importance visualization dashboard.
* Incorporate multi-year trend modeling (2018–2022).

---

### **Author**

**Atharva Sharma**
B.Tech, Mechanical Engineering — IIT Ropar
Software Developer Intern | Data Science & ML Enthusiast
[GitHub](https://github.com/atharva729) | [Kaggle](https://www.kaggle.com/atharva729)
