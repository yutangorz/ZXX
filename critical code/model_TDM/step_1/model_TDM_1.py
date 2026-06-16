import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

excel_file_path = 'TDM_all_feature.xlsx'
df = pd.read_excel(excel_file_path)
data = df.copy()

data['TDM'] = data['TDM'].apply(lambda x: 1 if x != 0 else 0)

drop_cols = ['formula', 'TDM']
X = data.drop(drop_cols, axis=1)
y = data['TDM'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = xgb.XGBClassifier(
    objective='binary:logistic',
    colsample_bytree=0.7,
    gamma=0.5,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=1,
    n_estimators=1500,
    subsample=0.8,
    reg_lambda=0.8,
    reg_alpha=0.3,
    scale_pos_weight=3,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:")
print(report)
print("\nConfusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Zero TDM value', 'Non-zero TDM value'], yticklabels=['Zero TDM value', 'Non-zero TDM value'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix of the TDM classification model')
plt.show()

model.save_model('model_binary_tdm.json')

results = pd.DataFrame({'True Value': y_test, 'Predicted Value': y_pred, 'Predicted Probability': y_pred_prob})

results.to_csv('binary_tdm.csv', index=False)