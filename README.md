# HousePricePrediction

## Overall Approach
The overall approach to this project was as follows:-

1) Examine data and look for feature engineering opportunities, then research Kaggle for additional ideas
(see Acknowledgements section below).

2) Form a baseline model using Linear Regression on (close to) the raw data i.e. with just the
Garage_Yr_Blt NA values set to the year the house was built, and the categorical variables
transformed to dummy variables.

3) A selection of other models was also chosen (Ridge, Lasso, Random Forest, Gradient Boosting
Regression Trees from scikit-learn, and also XGBoost) with the view to get experience with all and to
choose the best two models at the end.

4) These baseline models (with default parameters) were then run on a 70% split and the 10-fold crossvalidated
results were recorded.

5) Various feature engineering options were then tried on this 70/30 split and all the baseline models run
for each option. If a significant number of the models showed improvement then the feature or
transformation was kept otherwise it was rejected.

6) Each models parameters were then tuned using a combination of Grid Search for Lasso and Ridge, and
hand-tuning for the GBM. Additionally the scikit-optimize library was used to perform an ‘intelligent’
search of the parameter spaces (Bayesian optimization using Gaussian processes) to check the handtuning
hadn’t found a local minima. Despite 12+ hours of run-time, the hand-tuned / grid search
models performed slightly better than the scikit-optimize approach.

7) The various feature engineering options identified in 5 above were then reapplied, checking that they
still made a significant improvement in the modelling (they did).

8) The best two models were chosen based on mean performance and also consistency of performance.
For a final check of overall performance, these models were also run on the 10 sets of test ids provided
and the RMSEs recorded. A simulation was then run to calculate the chance of achieving a mean
RMSE on 3 of the 10 sets, of less than 0.12 (shown in Figure 3 in the Appendix below).

The technology used for this project was Python, Jupyter notebooks, with Scikit-learn and XGBoost libraries.
The modeling was run on an iMac i7 4.2GHz, 40GB and the runtime of the Python code is 25 seconds. The
metric used throughout this report is the 10-fold cross-validated RMSE on the log Sale_Price prediction error.

## Feature Engineering
The following Feature Engineering step were found to be beneficial enough to include in the final modelling:

1) Garage_Yr_Blt NA values were updated to the house Year_Built values based on the fact that most
garages would be built when the house was built (and also the lack of any other specific information!)

2) Drop the Utilities, and Condition_2 columns as they are almost all a single value.

3) Combine rare values (< 3 occurrences) in categorical columns to a single dummy value.

4) The following variables were changed from numeric to categorical based on the fact there may be a
non-linear relationship between the categorical values (Bsmt_Full_Bath, Bsmt_Half_Bath, Full_Bath,
Half_Bath, Bedroom_AbvGr, Kitchen_AbvGr, TotRms_AbvGrd, Fireplaces, Garage_Cars)

5) A numeric variable was created for each categorical variable, and it’s value was set to the mean sale
price for the particular category value.

6) Approximately 20 new variables were created as derived from the existing variables, summing some
key values and in other cases multiplying as appropriate to create meaningful measurements that were
judged to affect house price (for example, zTotalPlace was the sum of 8 other areas). The initial list
was obtained as described in the Acknowledgements section, however it was refined and added to.

7) For columns that are highly skewed (>5), a log transformation was applied to improve normality.

8) Perform PCA on the training data and add the first 5 principal components to the train and test

## Results
The results of the above feature engineering steps are shown in Figure 1 below, where almost every model
improved at almost every step. The notable exception is that Ridge did not respond well to adding new
columns, but responded very well to log transform of high skew variables. The linear regression benchmark
is shown only on (almost) raw data, due to collinearity on the feature engineered data causing problems.

## Conclusion
On this dataset, gradient boosted models required far less feature engineering than the linear models, so their
performance improved more with tuning than it did with feature engineering, and the tuning was time
consuming. Conversely the linear models performance improved more with feature engineering than with
tuning, however the tuning was much simpler than with the gradient boosted models.

The final results were that the linear regression benchmark on the raw data was significantly outperformed by
all other models. Random forest also performed relatively poorly. However Ridge, Lasso and ElasticNet all
performed quite well, and were only outperformed by scikit-learn’s gradient boosted regressor and XGBoost.
Somewhat surprisingly, XGBoost was not quite the best performer as may expected from its reputation,
however perhaps this was due to parameter tuning.

The two models chosen for submission were scikit-learn Gradient Boosted Regressor (GBR) and XGBoost.
These two models performed better than the linear models, though ElasticNet was not far behind and would
have been a reasonable choice if the model families were required to be different.

Both the GBR and XGBoost models achieved 10-fold cross-validated RMSE of less than 0.120 (as shown in
Figure 2 above). And on the 10 data splits provided for this project, GBR obtained an average RMSE of
0.11338, and XGBoost obtained and average RMSE of 0.11486 (as shown in the appendix) – also both less
than the 0.120 required.
