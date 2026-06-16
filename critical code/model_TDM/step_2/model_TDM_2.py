#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
import os

excel_file_path = 'TDMs_less_feature.xlsx'
df = pd.read_excel(excel_file_path)

delect_col = ['formula', 'TDMs']
X = df.drop(delect_col, axis=1)
y = df['TDMs'].values

best_params = dict(
    objective='reg:squarederror',
    learning_rate=0.07,
    max_depth=6,
    min_child_weight=5,
    subsample=0.75,
    colsample_bytree=0.7,
    reg_alpha=1.2,
    reg_lambda=1.8,
    gamma=0.25,
    n_jobs=-1,
    random_state=42
)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
best_iters, rmses, r2s = [], [], []

for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y), 1):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model = xgb.XGBRegressor(
        **best_params,
        n_estimators=8000,
        early_stopping_rounds=150,
        eval_metric='rmse'
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    best_iters.append(model.best_iteration)

    y_pred = model.predict(X_val)
    r2s.append(r2_score(y_val, y_pred))
    rmses.append(np.sqrt(mean_squared_error(y_val, y_pred)))

opt_rounds = int(np.mean(best_iters))
print(f'CV: {opt_rounds}, R2: {np.mean(r2s):.5f}, RMSE: {np.mean(rmses):.5f}')

final_model = xgb.XGBRegressor(**best_params, n_estimators=opt_rounds)
final_model.fit(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

y_train_pred = final_model.predict(X_train)
y_test_pred  = final_model.predict(X_test)

test_scatter_df = X_test.copy()
test_scatter_df['y_true']  = y_test
test_scatter_df['y_pred']  = y_test_pred
test_scatter_df.to_csv('test_scatter_data.csv', index=False, encoding='utf-8-sig')

def plot_scatter(y_true, y_pred, title, save_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_true, y_pred, c='blue', alpha=0.6, s=28, edgecolors='k', linewidths=0.2)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'k--', lw=1.5)
    r2  = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse= np.sqrt(mean_squared_error(y_true, y_pred))
    ax.text(0.05, 0.92, f'R² = {r2:.5f}\nMAE = {mae:.5f}\nRMSE = {rmse:.5f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.25))
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('True Values', fontsize=11)
    ax.set_ylabel('Predicted Values', fontsize=11)
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

plot_scatter(y_train, y_train_pred, 'Training Set', 'scatter_train.png')
plot_scatter(y_test,  y_test_pred,  'Test Set',   'scatter_test.png')

pd.DataFrame({'y_true': y_test, 'y_pred': y_test_pred}).to_csv('predictions.csv', index=False)
final_model.save_model('best_model.xgb')
print("Model saved as best_model.xgb")

print("Calculating SHAP values...")
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X)

plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig('shap_summary_beeswarm.png', dpi=300)
plt.close()

plt.figure()
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('shap_bar_importance.png', dpi=300)
plt.close()

abs_mean = np.abs(shap_values).mean(axis=0)
feat_imp = pd.DataFrame({
    'feature': X.columns,
    'shap_abs_mean': abs_mean
}).sort_values('shap_abs_mean', ascending=False)
feat_imp.to_csv('feature_importance.csv', index=False, encoding='utf-8-sig')

print("SHAP analysis complete.")