# Rainfall Prediction

This is a classification model development exrcise for binary predictions (0/1, hot/cold, good/bad) - in this case 0/1 where we predict whether it would rain or not.

## My approach to the Problem

1. **Data Loading and Initial Exploration**:
   - Loaded the training, test, and submission datasets using `pandas`.
   - Displayed the first few rows of the submission dataset to understand its structure.

2. **Model Evaluation Function**:
   - Defined a function `evaluate_binary_model` to evaluate binary classification models. This function calculates various metrics such as accuracy, precision, recall, F1 score, ROC AUC, and Brier score. It also plots the confusion matrix, ROC curve, and reliability diagram.

3. **Feature Importance Visualization**:
   - Created a function `plot_feature_importance` to visualize the importance of features used in the model. This function supports models that have either `feature_importances_` or `coef_` attributes.

4. **Data Exploration**:
   - Displayed the first few rows and summary statistics of the training dataset to get an initial understanding of the data.
   - Checked for missing values and the distribution of the target variable `rainfall`.

5. **Exploratory Data Analysis (EDA)**:
   - Conducted EDA to check for missing values, basic statistics, and correlations.
   - Visualized the distribution of rainfall frequency by the day of the year using a box plot.

6. **Model Training and Evaluation**:
   - The script does not explicitly show the model training process, but it sets up the necessary functions and data exploration steps that would typically precede model training and evaluation.

7. **Conclusion**:
   - The approach involves thorough data exploration, setting up evaluation metrics, and preparing the data for model training. The script is well-structured to handle binary classification tasks, particularly focusing on evaluating model performance and understanding feature importance.

## How to use this:

Launch your terminal and run

```cmd
# Creat Virtual environment
python -m venv <venv_name> # replace <venv_name> with your desired name
<venv_name>\Scripts\activate # activate
```

```bash
# Clone repository
git clone https://github.com/AkanimohOD19A/tidyTuesday-rainfall.git

# Install the requirements
pip install -U -r requirements.txt
```

Either convert to script with `jupyter nbconvert --to script xgbclassifier-accuracy-0-8658.ipynb` and run as script or run the ipynb - it's up to you.