# Assignments

## Case Studies

-   BankSecure Example
-   Bank Marketing
-   Dow Jones Case Study
-   Customer Retention
-   Cancer Detection

## Project

-   Proposals
-   Presentation

### Case Study Checklist

**Part A – Data Understanding & Prep**

-   Load CSV (read.csv2(..., sep=";") since it’s semicolon-delimited).

-   Check structure (str(), summary(), table(y)).

-   Explore imbalance (expected: \~10–12% “yes”).

-   Clean/transform variables (factor encoding, maybe bin age, handle “unknown”s).

**Part B – Modeling**

-   Train/test split (e.g., 70/30).

-   Fit Logistic Regression and LDA.

-   Optionally add Decision Tree / Random Forest for comparison.

-   Handle imbalance → try oversampling (SMOTE), undersampling, or class weights.

-   Compare models: AUC, accuracy, sensitivity, specificity.

**Part C – Insights**

-   Feature importance (coefficients, standardized loadings, variable importance plots).

-   Who are the most likely responders? (e.g., older customers, certain job types, people contacted in certain months).

**Part D – Strategy & Recommendations**

-   If only contacting 20% of customers:

-   Rank by predicted probability.

-   Estimate expected lift (compare to random 20%).

-   Discuss risks of bias (e.g., excluding “unknown education” group).

-   Deliverables

**Case Study Report**

Word/PDF with Executive Summary → Problem → Data → Methods → Results → Recommendations
