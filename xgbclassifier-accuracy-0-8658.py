#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

sub_df = pd.read_csv("data/sample_submission.csv")
sub_df.head().T


# In[5]:


from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            roc_auc_score, confusion_matrix, classification_report,
                            mean_squared_error, r2_score, brier_score_loss)
from sklearn.calibration import calibration_curve

def evaluate_binary_model(model, X_test, y_test):
    # Get predictions and probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for positive class

    # Classification metrics
    print("Classification Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Brier Score: {brier_score_loss(y_test, y_prob):.4f}")  # Lower is better

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Rain', 'Rain'], 
                yticklabels=['No Rain', 'Rain'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ROC Curve
    from sklearn.metrics import RocCurveDisplay
    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.title('ROC Curve')
    plt.show()

    # Reliability diagram (calibration curve)
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Reliability Diagram')
    plt.show()

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_prob),
        'brier': brier_score_loss(y_test, y_prob)
    }


# Feature importance visualization
def plot_feature_importance(model, feature_names):
    # Get feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        print("Model doesn't have feature_importances_ or coef_ attributes")
        return

    # Sort feature importances
    indices = np.argsort(importances)[::-1]

    # Plot
    plt.figure(figsize=(10, 8))
    plt.title('Feature Importances')
    plt.bar(range(len(indices)), importances[indices], align='center')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()


# In[6]:


train_df.head()


# In[7]:


train_df.describe()


# In[8]:


train_df.info()


# In[9]:


train_df.rainfall.value_counts()


# ## Exploratory Data Analysis
# - Check for missing columns/values
# - Basic stats/distro.
# - Correlations, Feature VIP & Engineering
# 
# From the info. output there seems to be no missing value - moving on..

# In[10]:


# Distribution of rainfall
plt.figure(figsize = (12,6))
sns.boxplot(data=train_df, x="rainfall", y="day")#, hue="alive")
plt.xlabel("Days of the Year")
plt.ylabel("Rainfall Frequency")
plt.title("Average Rainfall frequency by the Day of Year")
plt.show()


# There appears to be no significant difference in the distribution of rainfall frequency between days with and without rain.
# 
# This visualization seems unusual, as typically we would expect to see a clearer difference between rainy and non-rainy days.

# In[11]:


# Day x Rainfall
grouped_days = train_df.groupby(['day']).agg({'rainfall': 'mean'}).reset_index()

plt.figure(figsize = (12,6))
sns.lineplot(x="day", y="rainfall", data = grouped_days)
plt.xlabel("Days of the Year")
plt.ylabel("Rainfall Frequency")
plt.title("Average Rainfall frequency by the Day of Year")
plt.show()


#  - The rainfall frequency appears highly variable throughout the year, with frequent fluctuations between higher and lower values.
#  - Most of the year shows relatively high rainfall frequencies (between 0.6-1.0), suggesting that rain is common on many days.
#  - There are notable drops in rainfall frequency at various points throughout the year. The most significant drop appears around day 150, where the frequency approaches 0, indicating a period with very little chance of rainfall.
#  - Several other low points occur around days 25-50 and 175-225, which might represent drier seasons or periods.
#  - The end of the year (days 300-365) shows consistently high rainfall frequency with some fluctuations, suggesting a generally wet period.
# 
# 
#  We can still this do this for every qunlitative variable in the dataset - however, let's just check for general correlation for determine the higher influential variables.
# 
# 
#  ##### Correlation

# In[12]:


corr_matrix = train_df.corr()

mask = np.zeros_like(corr_matrix, dtype = np.bool_)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize = (11,9))
sns.heatmap(corr_matrix, mask = mask, center = 0,
            square = True, linewidths=.5,
            cbar_kws = {"shrink": .9}, vmin = -1.2,
            vmax = 1.2, cmap = "coolwarm", annot = True)


# **Key Correlations with Rainfall:**
# 
# Strong positive correlations:
# 
# Cloud cover (0.45): More clouds correlate with higher rainfall
# Sunshine (-0.56): Strong negative correlation, indicating less sunshine associates with more rain
# 
# 
# **Weak or negligible correlations:**
# 
# Temperature variables (temperature, maxtemp, mintemp) show very weak negative correlations with rainfall
# Humidity has a small positive correlation (0.082)
# Wind direction and speed show minimal correlation
# 
# Here's what we are going to do - 
# 
# we will create a bunch of intermediate variables in _feature-engineering_, using the correlation here as guidance, following which we will handle redundancies and then run a _PCA_ for the VIP features for *Model Development*.

# In[13]:


train_df.head().T


# In[14]:


## Feature Engineering
df = train_df.sort_values(by = 'day').copy()

df['cloud_humidity'] = df['cloud'] * df['humidity']
df['dew_temp_diff'] = df['temparature'] - df['dewpoint']

# Temporal features
df['pressure_3d_avg'] = df['pressure'].rolling(window=3).mean()
df['humidity_7d_avg'] = df['humidity'].rolling(window=7).mean()

# Pressure Gradient
df['pressure_change'] = df['pressure'].diff()

# Cyclical features for day of year
df['temp_dewpoint_spread'] = df['temparature'] - df['dewpoint']

# Pressure change indicators
df['pressure_rising'] = (df['pressure'].diff() > 0).astype(int)
df['pressure_system'] = df['pressure'].diff().apply(
    lambda x: 'rising' if x > 0.01 else (
        'falling' if x < -0.01 else 'stable'
    )
)

## Cloud buildup rate
df['cloud_buildup'] = df['cloud'].diff()

# Threshold-based features (corrected syntax)
df['humidity_high'] = (df['humidity'] > 70).astype(int)
df['dewpoint_depression_critical'] = ((df['temparature'] - df['dewpoint']))

## Lagged variables
df['rainfall_lag1'] = df['rainfall'].shift(1)
df['pressure_lag1'] = df['pressure'].shift(1)
df['cloud_change'] = df['cloud'] - df['cloud'].shift(1)

### domain-specific indices
# Simple CAPE approximation
df['simple_cape'] = df['humidity'] * df['temp_dewpoint_spread'] * (df['temparature'] > 20).astype(int)

# Storm Potential
df['storm_index'] = (df['humidity']/100) * df['cloud'] * (1 - df['pressure']/1013.25)


# In[15]:


df.head().T


# In[16]:


df.pressure_system.value_counts()


# In[17]:


## Backward Fill
cols_to_fill = ['pressure_3d_avg', 'humidity_7d_avg', 'pressure_change', 'cloud_buildup', 
               'rainfall_lag1', 'pressure_lag1', 'cloud_change']

df[cols_to_fill] = df[cols_to_fill].fillna(method = 'bfill')


# In[18]:


df.info()


# In[19]:


## Feat. Normalization

num_features = df.select_dtypes("float64").columns.to_list()
scaler = StandardScaler()
df[num_features] = scaler.fit_transform(df[num_features])


# Now, let's check for Feature Importance - the VIPs

# In[20]:


# a lil. more pre-processing
encoder = OneHotEncoder(sparse_output=False)

ohe = encoder.fit_transform(df[['pressure_system']])
ohd = pd.DataFrame(ohe, columns=encoder.get_feature_names_out(['pressure_system']))

df = pd.concat([df, ohd], axis = 1)
df.drop(columns = 'pressure_system', inplace = True)


# In[21]:


X = df.drop(columns = "rainfall")
y = df['rainfall']

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 234)

# model
rf_model = RandomForestRegressor()
rf_model.fit(
    X_train, y_train
)

# feat. importance
importances = pd.DataFrame(
    {
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }
).sort_values('importance', ascending=False).reset_index()

importances


# Wow - did not expect that _cloud-humidity_ but it comes out on top - the combination of - *Cloud* × *Humidity*: Could capture conditions favorable for precipitation. Let's add more interacting variables.

# In[22]:


df['cloud_sunshine'] = df['cloud'] * df['sunshine']
df['humidity_pressure'] = df['humidity'] * df['pressure_change']
df['cloud_pressure_change'] = df['cloud'] * df['pressure_change']
df['sunshine_humidity'] = df['sunshine'] * df['humidity']

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(2, include_bias = False)
top_features = ["cloud_humidity", "sunshine", "cloud", 
                "windspeed", "pressure_change",
                "pressure_lag1", "dewpoint"]

model_features = top_features + ["cloud_sunshine", "humidity_pressure"]
poly_features = poly.fit_transform(df[top_features])


# In[24]:


from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits = 5)
model = XGBClassifier(
    learning_rate = 0.05,
    n_estimators = 200,
    max_depth = 5,
    subsample=0.8,
    coolsample_bytree=0.8
)

for train_idx, test_idx in tscv.split(df[model_features]):
    X_train, X_test = df.iloc[train_idx][model_features], df.iloc[test_idx][model_features]
    y_train, y_test = df.iloc[train_idx]['rainfall'], df.iloc[test_idx]['rainfall']
    model.fit(X_train, y_train)

evaluate_binary_model(model, X_test, y_test)



#  - Accuracy (0.8658): This indicates that 86.58% of the predictions made the model are correct. It's a general measure of how often the model is right.
#  - Precision (0.9125): This tells you that when the model predicts a positive class, it is correct 91.25% of the time. Precision is important when the cost of false positives is high.
#  - Recall (0.9218): This means that the model correctly identifies 92.18% of all actual positive cases. Recall is crucial when the cost of false negatives is high.
#  - F1 Score (0.9171): This is the harmonic mean of precision and recall, providing a balance between the two. An F1 score of 0.9171 indicates a good balance between precision and recall.
#  - ROC AUC (0.8638): The ROC AUC score of 0.8638 shows that the model has a good ability to distinguish between the positive and negative classes. A score closer to 1 indicates better performance.
#  - Brier Score (0.1030): This measures the accuracy of probabilistic predictions. A lower Brier score indicates better calibrated predictions. A score of 0.1030 suggests that the model's probability estimates are quite accurate.
# 
#  Now - let's validate with the actual test records..

# In[46]:


test_ids = train_df['id'].copy
test_features = test_df.drop(columns=['id'])


# In[48]:


## Validate Model
# feat. engineering function

def feat_eng(df: pd.DataFrame):
    df = df.sort_values(by = 'day').copy()

    df['cloud_humidity'] = df['cloud'] * df['humidity']
    df['dew_temp_diff'] = df['temparature'] - df['dewpoint']

    # Temporal features
    df['pressure_3d_avg'] = df['pressure'].rolling(window=3).mean()
    df['humidity_7d_avg'] = df['humidity'].rolling(window=7).mean()

    # Pressure Gradient
    df['pressure_change'] = df['pressure'].diff()

    # Cyclical features for day of year
    df['temp_dewpoint_spread'] = df['temparature'] - df['dewpoint']

    # Pressure change indicators
    df['pressure_rising'] = (df['pressure'].diff() > 0).astype(int)
    df['pressure_system'] = df['pressure'].diff().apply(
        lambda x: 'rising' if x > 0.01 else (
            'falling' if x < -0.01 else 'stable'
        )
    )

    ## Cloud buildup rate
    df['cloud_buildup'] = df['cloud'].diff()

    # Threshold-based features (corrected syntax)
    df['humidity_high'] = (df['humidity'] > 70).astype(int)
    df['dewpoint_depression_critical'] = ((df['temparature'] - df['dewpoint']))

    ## Lagged variables
    # df['rainfall_lag1'] = df['rainfall'].shift(1)
    df['pressure_lag1'] = df['pressure'].shift(1)
    df['cloud_change'] = df['cloud'] - df['cloud'].shift(1)

    ### domain-specific indices
    # Simple CAPE approximation
    df['simple_cape'] = df['humidity'] * df['temp_dewpoint_spread'] * (df['temparature'] > 20).astype(int)

    # Storm Potential
    df['storm_index'] = (df['humidity']/100) * df['cloud'] * (1 - df['pressure']/1013.25)

    cols_to_fill = ['pressure_3d_avg', 'humidity_7d_avg', 'pressure_change', 'cloud_buildup', 
               'pressure_lag1', 'cloud_change']

    df[cols_to_fill] = df[cols_to_fill].fillna(method = 'bfill')

    num_features = df.select_dtypes("float64").columns.to_list()
    scaler = StandardScaler()
    df[num_features] = scaler.fit_transform(df[num_features])

    encoder = OneHotEncoder(sparse_output=False)

    ohe = encoder.fit_transform(df[['pressure_system']])
    ohd = pd.DataFrame(ohe, columns=encoder.get_feature_names_out(['pressure_system']))

    df = pd.concat([df, ohd], axis = 1)
    df.drop(columns = 'pressure_system', inplace = True)

    # model features
    df['cloud_sunshine'] = df['cloud'] * df['sunshine']
    df['humidity_pressure'] = df['humidity'] * df['pressure_change']
    df['cloud_pressure_change'] = df['cloud'] * df['pressure_change']
    df['sunshine_humidity'] = df['sunshine'] * df['humidity']

    poly = PolynomialFeatures(2, include_bias = False)
    top_features = ["cloud_humidity", "sunshine", "cloud", 
                    "windspeed", "pressure_change",
                    "pressure_lag1", "dewpoint"]

    model_features = top_features + ["cloud_sunshine", "humidity_pressure"]
    poly_features = poly.fit_transform(df[top_features])

    return df, top_features, model_features, poly_features

df_test, top_f, model_f, poly_f = feat_eng(test_df)


# In[ ]:


# #  Ensure alignment
# if len(df_test) != len(test_df):
#     print(f"Transformed Shape: {len(df_test)}", "\n")
#     print(f"Original Shape: {len(test_df)}", "\n")
#     raise ValueError("Processed test data has a "
#     "different number of rows than the original")

assert len(df_test) == len(test_features), "Row count mismatch after processing"


# In[61]:


test_X = df_test[model_f]

predictions = model.predict(test_X)
y_proba = model.predict_proba(test_X)[:, 1]

# 1. Check dimensions
print(f"Original test_df shape: {test_df.shape}")
print(f"Processed df_test shape: {df_test.shape}")
print(f"Predictions shape: {predictions.shape}")
print(f"Number of IDs: {len(test_df['id'])}")

# 2. Create a submission DataFrame
submission_df = pd.DataFrame({
    'id': test_df['id'].values,
    'rainfall': predictions
})

# 3. Verify submission_df
print(f"Submissio shape: {submission_df.shape}")
print(submission_df.head())

# 6. Save submission
submission_df.to_csv('submission.csv', index=False)
print("\nCreated Submission File")

