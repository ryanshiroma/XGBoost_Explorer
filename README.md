# XGBoost Explorer (A simple 1-dimensional XGBoost fitter visualizer)
All too often as busy financial data scientists, we spend 90% of our time cleaning and prepping data, another 20% modeling, and another 20% justifying our choices to our business partners and validators/regulators. In an attempt to better understand how our choices on the most common XGBoost regularization hyperparameters, I'm working on a small demo to visualize how small changes in some hyperparameter settings can affect tree fitting.

Currently, I've implemented the following hyperparameters:
- `max_depth`
- `learning_rate`
- `min_child_weight`
- `lambda` (l2 regularization)

On my to-do list is:
- `subsample`
- `gamma` (min split loss)
- `alpha` (l1 regularization)


Other things on the to-do list:
 - Add more documentation (theorywise and codewise)
 - Convert the code for use on AWS
 - Add more loss functions
 - Add a setting to pick between Gradient Descent(standard GBM) and Newton Optimization (XGBoost/LightGBM among others)
 - Add a plot showing potential test/error curves as new boosters are added.

## Quick Start Guide
```
conda env create -f environment.yml
conda activate xgboost_explorer
python xgboost_explorer.py
```
Navigate to `127.0.0.1:8050`

1) **First begin by selecting your data settings**
- **Problem Type**: `classification` for binary outcomes and `regression` for continuous
- **Function**:  setting to pick bewtween multiple functions to fit on
- **Noise**: the amount of Gaussian noise to add to the dataset where the variance is the slider value divided by 75. (no particular reason for 75 other than that it made the slider number on a reasonable scale) 
- **Sample Size**: the number of data points to generate
- **Loss Function**: setting to switch between which loss function to minimize, MSE or logloss

The model plot(top left) shows the data points(blue dots), the true model function(blue line) and the current fitted model(red line).
The booster plot(bottom left) shows the pseudo-residuals at the current tree(when you start the psuedo-residuals are just the datapoints themselves).

2) **Play around with the tree settings and see how changing the tree settings affect the fit**
The booster plot will now show blue vertical lines where the tree split points have been made. The green subplot underneath show the loss at that particular split point. The red lines represent the predicted value, "weight", for the observations in that node. The right plot will update with the current tree fitted to the pseudo-residuals. Lastly, the model plot will update with a peek at how this booster will contribute to the final model, shown with the filled in red area.
3) **Click Save Booster to "boost" the current full model with your new booster**
The booster will now be added to the full model and the new pseudo-residuals will be shown in the model plot. If you are running a regression, pay particular attention to how the pseudo-residuals shift up or down as new boosters are added.
4) **Rinse, Repeat** 

![dashboard screenshot](https://github.com/ryanshiroma/XGBoost_Explorer/blob/master/xgboost_explorer_dashboard.png)
