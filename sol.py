# Importing required libraries and loading data
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the dataset
crops = pd.read_csv("soil_measures.csv")

# Checking if our dataset has empty values.
null_vals = np.where(pd.isnull(crops))

print(null_vals)




## Analysing the features
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

scaler = StandardScaler()
kf = KFold(n_splits = 5)


y = crops['crop'].values
X = crops.drop(['crop'], axis=1).values

label_encoder = LabelEncoder()
label_encoder.fit(y)
y = label_encoder.transform(y)

y.reshape(-1,1)

#print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_train_scaled = scaler.fit_transform(X_train)


params = {'alpha' : np.linspace(0,1,20)}

lasso = Lasso()
lasso_cv=GridSearchCV(lasso, param_grid=params, cv=kf)
lasso_cv.fit(X_train_scaled, y_train)

print(lasso_cv.best_params_)

lasso = Lasso(alpha=lasso_cv.best_params_['alpha'])
lasso.fit(X_train_scaled, y_train)

plt.bar(crops.drop(['crop'], axis=1).columns.values, np.abs(lasso.coef_))
plt.show





## Preprocessing the data and doing some LogisticRegression
scaler = StandardScaler()
kf = KFold(n_splits = 5)


y = crops['crop'].values
X = crops['P'].values

label_encoder = LabelEncoder()
label_encoder.fit(y)
y = label_encoder.transform(y)
X = X.reshape(-1,1)
y = y.reshape(-1,1)

print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

steps = [('scaler', StandardScaler()), ('logreg', LogisticRegression())]
params = {'logreg__C' : np.linspace(0,1, 20)}
pipeline = Pipeline(steps)
log_cv = GridSearchCV(pipeline, param_grid = params, cv=kf)
log_cv.fit(X_train, y_train)
#scores = cross_val_score(pipeline, X_train, y_train, cv=kf)





steps = [('scaler', StandardScaler()), ('logreg', LogisticRegression(C=log_cv.best_params_["logreg__C"]))]
pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)
scores = pipeline.score(X_test, y_test)
print(pipeline.score(X_test, y_test))




## Further data analysis

# Read in our dataset
df = pd.read_csv("soil_measures.csv")

# Check for missing values
missing_vals = df.isnull().values.any()
if missing_vals == True:
    print("Our dataset has missing values")
else:
    print("No missing values!")

# Check for crop types

print(df['crop'].unique())

# Split data into sample features and label

# y = df['crop'].values
# X = df.drop(['crop'], axis=1).values


# le = LabelEncoder()
# le.fit(y)
# y = le.transform(y)

# print(np.unique(y))

# Train Test Split

# X_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.4, random_state = 0)

df = pd.read_csv("soil_measures.csv")
y = crops['crop'].values
label_encoder = LabelEncoder()
label_encoder.fit(y)
y = label_encoder.transform(y)
y = y.reshape(-1, 1)

features_dict = {feature: 0 for feature in df.columns.values if feature != "crop"}
for feature in features_dict:
    X = crops[feature].values
    X = X.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    standardscaler = StandardScaler()
    logreg = LogisticRegression(multi_class='multinomial')
    X_train_scaled = standardscaler.fit_transform(X_train, y_train)
    logreg.fit(X_train_scaled, y_train)

    X_test_scaled = standardscaler.transform(X_test)
    y_pred = logreg.predict(X_test)
    features_dict[feature] = f1_score(y_test, y_pred, average='weighted')

print([f"F1-score for {feature}: {features_dict[feature]}" for feature in features_dict])

best_predictive_feature = {max(features_dict): features_dict[max(features_dict)]}



## Finding best predictive feature
# Read the data into a pandas DF and perform exploratory dat aanalysis
df = pd.read_csv("soil_measures.csv")

# Check for missing values
missing_vals = df.isna().values.sum()
if missing_vals != 0:
    raise ValueError

# Check for crop types
class_labels = df['crop'].unique()

# Features and target variables
# Create a variable contating the features, and class labels

y = df['crop']
X = df.drop(['crop'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Evaluate performance


features_dict = dict()
feature_performance = dict()
for feature in X.columns.values:
    log_reg = LogisticRegression(multi_class='multinomial')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[[feature]])
    log_reg.fit(X_train_scaled, y_train)
    X_test_scaled = scaler.transform(X_test[[feature]])
    y_pred = log_reg.predict(X_test_scaled)
    print(log_reg.score(X_test_scaled, y_test))
    feature_performance[feature] = f1_score(y_test, y_pred, average='weighted')

print(f"F1-score for {feature}: {feature_performance}")

best_predictive_feature = {'K': feature_performance['K']}

print(best_predictive_feature)




## Finding best predictive feature, second attempt
# All required libraries are imported here for you.
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the dataset
crops = pd.read_csv("soil_measures.csv")

# Check for missing values
crops.isna().sum()

# Check how many crops we have, i.e., multi-class target
crops.crop.unique()

# Split into feature and target sets
X = crops.drop(columns="crop")
y = crops["crop"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create a dictionary to store the model performance for each feature
feature_performance = {}

# Train a logistic regression model for each feature
for feature in ["N", "P", "K", "ph"]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[[feature]])
    log_reg = LogisticRegression(multi_class="multinomial")
    log_reg.fit(X_train_scaled, y_train)
    X_test_scaled = scaler.transform(X_test[[feature]])
    y_pred = log_reg.predict(X_test_scaled)

    # Calculate F1 score, the harmonic mean of precision and recall
    # Could also use balanced_accuracy_score
    f1 = metrics.f1_score(y_test, y_pred, average="weighted")

    # Add feature-f1 score pairs to the dictionary
    feature_performance[feature] = f1
    print(f"F1-score for {feature}: {f1}")

# K produced the best F1 score
# Store in best_predictive_feature dictionary
best_predictive_feature = {"K": feature_performance["K"]}
best_predictive_feature


