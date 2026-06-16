import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import os
import joblib

TRAIN_PATH = r"train_pca_features1.csv"
TEST_PATH = r"test_pca_features1.csv"
TARGET_COL = 'target'

OUTPUT_DIR = r"lightgbm_results"

LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'max_depth': 8,
    'learning_rate': 0.03,
    'n_estimators': 2000,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 10,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'verbose': -1,
    'random_state': 42
}

def get_writable_dir(preferred_dir):
    preferred = Path(preferred_dir)

    try:
        preferred.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"")
        return Path.cwd()

    test_file = preferred / ".write_test"
    try:
        test_file.write_text("test")
        test_file.unlink()
        return preferred
    except Exception as e:
        print(f"")
        fallback = Path.cwd()
        print(f"")
        return fallback

def train_and_predict():

    output_dir = get_writable_dir(OUTPUT_DIR)
    print(f"")

    print("=" * 60)
    print("=" * 60)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"")
    print(f"")

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL].values
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL].values
    feature_names = X_train.columns.tolist()

    print(f"")

    print("\n" + "=" * 60)
    print("=" * 60)

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    model = lgb.train(
        LGB_PARAMS,
        train_data,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )

    print(f"")

    print("\n" + "=" * 60)
    print("=" * 60)

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    print("\n" + "=" * 60)
    print("=" * 60)

    results_df = pd.DataFrame()
    results_df['sample_id'] = range(1, len(test_df) + 1)
    results_df['actual'] = y_test
    results_df['predicted'] = y_pred
    results_df['residual'] = y_test - y_pred
    results_df['abs_error'] = np.abs(y_test - y_pred)
    results_df['rel_error_pct'] = (np.abs(y_test - y_pred) / y_test * 100).round(2)

    for col in feature_names:
        results_df[f'feature_{col}'] = test_df[col].values

    table_path = output_dir / "prediction_results_table.csv"
    try:
        results_df.to_csv(table_path, index=False, encoding='utf-8-sig')
        print(f"")
    except Exception as e:
        print(f"")

    print(f"")
    print(f"")
    print(results_df[['sample_id', 'actual', 'predicted', 'abs_error']].head())

    print("\n" + "=" * 60)
    print("=" * 60)

    model_pkl_path = output_dir / "lightgbm_model_g.pkl"
    try:
        joblib.dump(model, model_pkl_path)
        print(f"")
    except Exception as e:
        print(f"")
        try:
            fallback_path = Path("lightgbm_model_g.pkl").absolute()
            joblib.dump(model, fallback_path)
            print(f"")
        except Exception as e2:
            print(f"")

    print("\n" + "=" * 60)
    print(f"")
    print("=" * 60)

    return results_df, model

if __name__ == "__main__":
    results_df, model = train_and_predict()