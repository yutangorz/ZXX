import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import os
import joblib

TRAIN_PATH = r"train_features2.csv"
TEST_PATH = r"test_features2.csv"
TARGET_COL = 'target'
OUTPUT_DIR = r"lightgbm_results"

LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

def get_writable_dir(preferred_dir):
    preferred = Path(preferred_dir)
    try:
        preferred.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error: {e}")
        return Path.cwd()
    test_file = preferred / ".write_test"
    try:
        test_file.write_text("test")
        test_file.unlink()
        return preferred
    except Exception as e:
        print(f"Error: {e}")
        fallback = Path.cwd()
        print(f"Using: {fallback}")
        return fallback

def train_and_predict():
    output_dir = get_writable_dir(OUTPUT_DIR)
    print(f"Output: {output_dir}")

    print("=" * 60)
    print("1. Load Data")
    print("=" * 60)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"Train: {train_df.shape[0]} samples, {train_df.shape[1] - 1} features")
    print(f"Test: {test_df.shape[0]} samples, {test_df.shape[1] - 1} features")

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL].values
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL].values
    feature_names = X_train.columns.tolist()

    print(f"Features: {len(feature_names)}")

    print("\n" + "=" * 60)
    print("2. Train LightGBM")
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

    print(f"Best iteration: {model.best_iteration}")

    print("\n" + "=" * 60)
    print("3. Predict")
    print("=" * 60)

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    print("\n" + "=" * 60)
    print("4. Results")
    print("=" * 60)

    results_df = pd.DataFrame()
    results_df['Sample_ID'] = range(1, len(test_df) + 1)
    results_df['Actual'] = y_test
    results_df['Predicted'] = y_pred
    results_df['Residual'] = y_test - y_pred
    results_df['AbsError'] = np.abs(y_test - y_pred)
    results_df['RelError(%)'] = (np.abs(y_test - y_pred) / y_test * 100).round(2)

    for col in feature_names:
        results_df[f'Feature_{col}'] = test_df[col].values

    table_path = output_dir / "prediction_results_table.csv"
    try:
        results_df.to_csv(table_path, index=False, encoding='utf-8-sig')
        print(f"Saved: {table_path}")
    except Exception as e:
        print(f"Error: {e}")

    print(f"Columns: {list(results_df.columns)}")
    print("\nPreview:")
    print(results_df[['Sample_ID', 'Actual', 'Predicted', 'AbsError']].head())

    print("\n" + "=" * 60)
    print("5. Save Model")
    print("=" * 60)

    model_pkl_path = output_dir / "lightgbm_model_f.pkl"
    try:
        joblib.dump(model, model_pkl_path)
        print(f"Saved: {model_pkl_path}")
    except Exception as e:
        print(f"Error: {e}")
        try:
            fallback_path = Path("lightgbm_model_f.pkl").absolute()
            joblib.dump(model, fallback_path)
            print(f"Saved: {fallback_path}")
        except Exception as e2:
            print(f"Error: {e2}")

    print("\n" + "=" * 60)
    print("Done!")
    print(f"Output: {output_dir}")
    print("=" * 60)

    return results_df, model

if __name__ == "__main__":
    results_df, model = train_and_predict()