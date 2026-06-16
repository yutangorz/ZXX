import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import DMatrix, train as xgb_train, XGBClassifier

data_df = pd.read_csv(r'bandtype_feature_class.csv')

le = LabelEncoder()
data_df['target'] = le.fit_transform(data_df['target'])
class_names = le.classes_.astype(str)

non_numeric_columns = data_df.select_dtypes(include=['object']).columns
if non_numeric_columns.size > 0:
    print("Warning: Found non-numeric columns:", non_numeric_columns)

    data_df = data_df.drop(columns=non_numeric_columns)

X = data_df.drop('target', axis=1)
y = data_df['target']

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    'max_depth': [3, 4],
    'eta': [0.01, 0.05],
    'subsample': [0.5, 0.7],
    'colsample_bytree': [0.5, 0.7],
    'lambda': [3.0, 5.0],
    'alpha': [0.3, 0.5],
    'gamma': [0.1, 0.3],
    'min_child_weight': [6, 8],
    'scale_pos_weight': [0.94]
}

xgb_clf = XGBClassifier(objective='binary:logistic', eval_metric='logloss', n_jobs=-1)
grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, scoring='f1_weighted', cv=5, n_jobs=-1, verbose=0)

nested_scores = cross_val_score(grid_search, X, y, cv=outer_cv, scoring='f1_weighted')

print('Nested CV F1 Score:', np.mean(nested_scores))

grid_search.fit(X, y)
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print('Best params:', best_params)
print('Best F1:', best_score)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'scale_pos_weight': 0.94
}
params.update(best_params)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

dtrain = DMatrix(X_train, label=y_train)
dtest  = DMatrix(X_test,  label=y_test)

model = xgb_train(params, dtrain, num_boost_round=2000,
                  evals=[(dtrain, 'train'), (dtest, 'test')],
                  early_stopping_rounds=150, verbose_eval=0)
model.save_model('xgboost_bt.json')

y_pred_train = (model.predict(dtrain) > 0.5).astype(int)
print('')
print('Acc: %.4f  F1: %.4f' % (accuracy_score(y_train, y_pred_train),
                               f1_score(y_train, y_pred_train)))
print(classification_report(y_train, y_pred_train, target_names=class_names))

y_pred_test = (model.predict(dtest) > 0.5).astype(int)
print('')
print('Acc: %.4f  F1: %.4f' % (accuracy_score(y_test, y_pred_test),
                               f1_score(y_test, y_pred_test)))
print(classification_report(y_test, y_pred_test, target_names=class_names))

def save_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

save_confusion_matrix(y_train, y_pred_train, 'Confusion Matrix (Training Set)', 'confusion_matrix_train.png')
save_confusion_matrix(y_test, y_pred_test, 'Confusion Matrix (Test Set)', 'confusion_matrix_test.png')

def save_roc_curve(y_true, y_prob, title, filename):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc_val = auc(fpr, tpr)

    print('AUC-ROC:', title, '%.4f' % roc_auc_val)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, lw=2, label='AUC = %.4f' % roc_auc_val)
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

y_prob_train = model.predict(dtrain)
y_prob_test = model.predict(dtest)

save_roc_curve(y_train, y_prob_train, 'ROC Curve (Training Set)', 'roc_curve_train.png')
save_roc_curve(y_test, y_prob_test, 'ROC Curve (Test Set)', 'roc_curve_test.png')