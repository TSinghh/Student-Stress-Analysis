import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

file_path = r"/content/StressLevelDataset.csv"
df = pd.read_csv(file_path)

missing_values = df.isnull().sum()
missing_values

numerical_columns = ['anxiety_level', 'self_esteem', 'depression', 'headache',
                      'blood_pressure', 'sleep_quality', 'academic_performance',
                      'social_support', 'peer_pressure', 'bullying', 'stress_level']

Q1 = df[numerical_columns].quantile(0.25)
Q3 = df[numerical_columns].quantile(0.75)
IQR = Q3 - Q1

df_clean = df[~((df[numerical_columns] < (Q1 - 1.5 * IQR)) | (df[numerical_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]

print(df_clean.shape)

descriptive_stats = df.describe()
descriptive_stats

correlation_matrix = df[numerical_columns].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Important Factors')
plt.show()

correlation_matrix = df.corr()
correlation_matrix

correlation_with_stress = correlation_matrix['stress_level'].sort_values(ascending=False)
print(correlation_with_stress)

correlation_threshold = 0.74
selected_features = correlation_with_stress[abs(correlation_with_stress) >= correlation_threshold].index
print(selected_features)

data_selected = df[selected_features]
print(data_selected.head())

correlation_matrix = df[selected_features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Important Factors')
plt.show()

#MODELLING:

X = data_selected.drop('stress_level', axis=1)  
y = data_selected['stress_level']  

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

y_pred_log = logistic_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))

log_cv_scores = cross_val_score(logistic_model, X_scaled, y, cv=5)
xgb_cv_scores = cross_val_score(xgb_model, X_scaled, y, cv=5)

print("Logistic Regression CV Accuracy: ", log_cv_scores.mean())
print("XGBoost CV Accuracy: ", xgb_cv_scores.mean())
