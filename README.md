## Flight Disruption Prediction System

**Introduction**

This project explores the prediction of flight delays and cancellations using machine learning techniques. It utilizes the publicly available Flight Delay Dataset (2018-2022) from Kaggle ([Flight Status Prediction](https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022)).
Link to the notebook: [Flight Disruption System.ipynb](https://www.kaggle.com/code/atharva729/flight-delay-prediction)

**Project Goals**

This project aims to:

- Train machine learning models to predict whether a flight will be cancelled or delayed.
- Develop a model to estimate the potential delay time for flights.

### What I did:
**Data Loading and Preprocessing:**

1. **Import Data:** The code reads the flight delay dataset from a CSV file using `pandas.read_csv`.
2. **Column Selection:** Specific columns relevant to flight delays (e.g., Origin, Destination, Departure Time, Arrival Time, Delay) are selected and assigned to a new DataFrame `df1`.
3. **Date Conversion:** The `FlightDate` column is converted to the day of the year using `pandas.to_datetime` and `dt.dayofyear`.
4. **Label Encoding:** Categorical columns like `OriginState`, `DestState`, and `Operating_Airline` are encoded using `LabelEncoder` to convert them into numerical representations. Optional: The original categorical columns can be dropped if no longer needed.
5. **Missing Value Handling:**
    - `ArrDelay` and `DepDelay` missing values are filled with the mean delay.
    - `DepartureDelayGroups` and `ArrivalDelayGroups` missing values are filled with 0.0 (assuming numerical representation for delay groups).
    - A custom function `impute_time_with_delay` is defined to impute missing `DepTime` and `ArrTime` based on corresponding scheduled times and delays, handling potential rollover past midnight.
    - `ActualElapsedTime` missing values are imputed by adding the average difference between `CRSElapsedTime` and delays (`DepDelay` and `ArrDelay`).

**Problem Definition and Data Splitting:**

1. **Problem Framing:** The code identifies three prediction problems:
    - **Cancellation:** Binary classification (flight cancelled or not)
    - **Arrival Delay:** Regression (predicting the arrival delay in minutes) with possible thresholding to categorize delays (e.g., no delay, minor delay, major delay).
    - **Departure Delay:** Similar to arrival delay prediction.
2. **Data Splitting:** Separate DataFrames (`X_cancel`, `y_cancel`, etc.) are created for each prediction task, containing features and target variables.
3. The `train_test_split` function is used to split each DataFrame into training, validation, and test sets for model training and evaluation.

**Logistic Regression for Cancellation Prediction:**

1. **Class Imbalance Handling:** The code addresses class imbalance (unequal distribution of cancelled vs. non-cancelled flights) using `RandomUnderSampler` from `imblearn` to undersample the majority class (non-cancelled).
2. **Feature Scaling:** `StandardScaler` from `sklearn.preprocessing` is used to standardize features (e.g., centering and scaling) for better model performance with Logistic Regression.
3. **Cross-Validation with Stratified KFold:**
    - `StratifiedKFold` ensures balanced class distribution in each fold.
    - The model is trained and evaluated on multiple folds (iterations) to assess generalizability.
    - Average cross-validation score is calculated.
4. **Model Training and Evaluation:**
    - A Logistic Regression model (`LogisticRegression` from `sklearn.linear_model`) is trained on the training data.
    - The model's performance is evaluated on the validation set using metrics like classification report and accuracy score.

**Hyperparameter Tuning for Logistic Regression:**

1. The code iterates through different values of the `max_iter` hyperparameter (maximum number of iterations) to find the best configuration for the Logistic Regression model.
2. The model is trained and evaluated on the test set for each `max_iter` value.
3. The results show that the model's accuracy increases with higher `max_iter` values, potentially leading to overfitting.

**Comparison with Stochastic Gradient Descent Classifier (SGDClassifier):**

1. The code trains an SGDClassifier for cancellation prediction and compares its performance with Logistic Regression.
2. The score fluctuates for the validation set with different `max_iter` values, potentially indicating sensitivity to hyperparameters.
3. Training on the whole dataset (without splitting) shows generally higher scores for SGDClassifier, but the results should be interpreted with caution because of potential overfitting without a dedicated test set.

**Scores**
The first run gave a score of 0.74 which was improved a lot to 0.99 (not overfit, I verified) using hyperparameter tuning.

### Delay Prediction

**Data Preparation for Departure Delay Prediction:**

1. **Feature Selection:** The code starts by selecting all features from `df1` except for `DepDelay` (target variable) itself.
2. **Column Removal:** It then removes several features related to arrival time, delay groups, cancellation, and diversion, assuming they're not relevant for predicting departure delays.

**Train-Test Split and Feature Scaling:**

1. **Splitting Data:** The code splits the remaining features (`X_dep`) and target variable (`y_dep`) into training and testing sets using `train_test_split`.
2. **Scaling Features (Optional):** Two attempts are made to train a Linear Regression model:
   - Without scaling (using `X_train` and `X_test` directly).
   - With feature scaling using `StandardScaler` (fitting the scaler on the training data and transforming both training and testing sets).

**Linear Regression Models:**

1. **Baseline Model:** A baseline Linear Regression model (`modelr`) is trained on the unscaled training data (`X_train`, `y_train`). It's evaluated on the test set (`X_test`, `y_test`), but the score (`modelr.score(X_test, y_test)`) is not shown (likely low due to potential lack of model fit).

**Regularized Linear Regression:**

1. **Ridge Regression:** The code implements Ridge Regression with an `alpha` parameter of 0.1 to introduce L2 regularization. This technique penalizes models with large coefficients, potentially reducing overfitting. The model is trained and evaluated, but the scores are not explicitly shown.
2. **Lasso Regression:** Similarly, Lasso Regression with an `alpha` parameter of 0.01 is implemented. Lasso applies L1 regularization, which can lead to sparse models by setting some coefficients to zero. The model is trained and evaluated, but the scores are not provided.

**Random Forest Regressor:**

1. **Data Augmentation:** An interesting approach is taken here. A temporary DataFrame (`temp`) is created by copying the training data (`X_train`) and adding a new column "DEPDELAY" containing the target variable (`y_train`).
2. **Subsampling:** A random sample (10% of the size) is drawn from `temp` using `.sample()`. This subsampled data (`X_forest`) is used to train a Random Forest Regressor (`forest`) with 100 estimators and a random state of 420. The model's performance on the test set is evaluated using `.score(X_test, y_test)`, but the score (likely negative) is not shown.

**Correlation Analysis:**

1. **Heatmap Creation:** The code calculates the correlation matrix for the entire DataFrame (`df1`) and creates a heatmap using Seaborn to visualize pairwise correlations between features.
2. **Low Correlation:** The heatmap likely shows that no feature has a strong correlation (above 0.1) with the departure delay (`DepDelay`).

**Evaluation Metrics:**

1. **Error Metrics:** After training an unseen model (not shown in the provided code), the code calculates various error metrics using the actual test values (`list(y_test)`) and predicted values (`pred`):
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Error (MAE)
   - R-squared (R2)
2. **Interpretation:** The provided scores (MSE, RMSE, MAE) are likely high, indicating a poor model fit. The negative R2 score suggests the model performs worse than predicting the average delay.

**Further Exploration Attempts:**

1. **Label Encoding:** The code explores label encoding for all non-numeric columns in the original DataFrame (`df`), suggesting a potential attempt to use categorical features for prediction. However, the specific usage of encoded features is not shown.

2. **Correlation with Encoded Features:** The correlation matrix is again calculated, likely for the DataFrame with encoded features (`df`), but the heatmap likely shows no significant correlations with departure delay.

3. **Feature Engineering:** The code attempts to create new features from existing ones by selecting a subset of `df` that excludes most real-time columns like departure and arrival times. A correlation matrix is calculated for this subset (`arr`), but again, no strong correlations are found.

4. **Model Training with Subset Features:** The code defines a new DataFrame (`dfdf`) containing only four features: `CRSDepTime`, `TaxiOut`, `WheelsOff`, and `TaxiIn`. A target variable (`target`) is created using the departure delay (`DepDelay`). A new temporary DataFrame (`dfdf`) is created by dropping rows with missing values

**Implementation Details**

The project is implemented in one Jupyter Notebook, linked here: [Flight Delay Prediction](https://www.kaggle.com/code/atharva729/flight-delay-prediction). This notebook provides a step-by-step walkthrough of the methodology described above.

**Dependencies**

This project requires the following Python libraries:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

**License**

This project is licensed under the Apache 2.0 License (see LICENSE file) for open-source usage and modification.

**Contributing**

Feel free to contribute to this project by creating pull requests with improvements, additional analyses, or different modeling approaches.

**Disclaimer**

This project is for educational purposes only. The models developed here are not guaranteed to be accurate in real-world flight delay prediction scenarios. Always refer to official airline information for the most up-to-date flight status updates.

