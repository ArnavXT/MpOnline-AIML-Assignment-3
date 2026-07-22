# Salary Prediction using Polynomial Regression
**Author:** Arnav Tripathi

**Registration Number:** 23BCE10756

**Application Number:** IN26011416

**Batch Number:** 2B

**Email ID:** arnav.23bce10756@vitbhopal.ac.in

## Objective
Build a Polynomial Regression model that predicts an employee's salary from their position level, capturing the non-linear relationship between the two (salaries rise slowly at junior levels and sharply at senior/executive levels).

## Dataset Link
Kaggle: [Position Salaries Dataset](https://www.kaggle.com/datasets/akram24/position-salaries)

## Libraries Used
pandas
numpy
matplotlib
scikit-learn

## Methodology
Data Understanding: Loaded the CSV with Pandas, inspected the first 5 rows, `.info()`, and `.describe()`. Identified `Level` as the input feature and `Salary` as the target (`Position` is a text label for `Level` and adds no modeling information).

Data Preprocessing: Confirmed there are no missing values, selected `X = Level`, `y = Salary`, and split the 10 rows into 80% train (8 rows) / 20% test (2 rows) with `random_state=42`.

Model Development: Transformed `Level` with `PolynomialFeatures(degree=3)` and fit a `LinearRegression` model on the transformed features (i.e., polynomial regression via a linear model over `[1, x, x^2, x^3]`).

Model Evaluation: Predicted salaries on the test split and scored the model with MAE, MSE, and R², plus a scatter plot of the raw data overlaid with the fitted polynomial curve.

## Results
**MAE:** ≈ 70,635

**MSE:** ≈ 6.26 × 10⁹

**R² Score:** ≈ 0.876

The regression curve (see notebook) closely follows the sharp upward bend in salary at senior levels, which a straight line cannot do. With only 10 total data points, the 80/20 split leaves just 2 test points, so these metrics are indicative rather than statistically robust — the largest error occurs at the topmost level (CEO), where the salary jump is steepest and data is sparsest.

## Conclusion
**Key findings:** Employee salaries grow non-linearly, starting modestly at junior levels and accelerating steeply at the executive level. Using a degree 3 Polynomial Regression model on an 80/20 train-test split captured this curvature well, producing a reasonable fit on the test data (R² ≈ 0.876).

**Difference between Linear and Polynomial Regression:** Linear Regression fits a straight line assuming a constant rate of change. In contrast, Polynomial Regression fits a non-linear curve by incorporating higher-degree terms of the feature (x², x³), allowing the model to bend and match complex data trends.

**Advantage of Polynomial Regression for this dataset:** The primary advantage is its ability to accurately follow the steep, accelerating salary growth for senior positions. A standard straight line would systematically under-predict or over-predict these salaries, whereas the polynomial curve closely matches the true non-linear relationship.
