import pandas as pd
import joblib
from pathlib import Path

print('dataset_exists', Path('PhiUSIIL_Phishing_URL_Dataset.csv').exists())

df = pd.read_csv('PhiUSIIL_Phishing_URL_Dataset.csv', nrows=5)
print('columns:', list(df.columns))
print('head:')
print(df.head().to_string())
print('dtypes:')
print(df.dtypes.to_string())

model_path = Path('model.pkl')
print('model_exists', model_path.exists())
if model_path.exists():
    model = joblib.load(model_path)
    print('model_type', type(model))
    if hasattr(model, 'get_params'):
        print('model_params_keys:', list(model.get_params().keys())[:20])
    if hasattr(model, 'named_steps'):
        print('pipeline_steps:', list(model.named_steps.keys()))
    if hasattr(model, 'feature_names_in_'):
        print('feature_names_in:', model.feature_names_in_)
    if hasattr(model, 'feature_names_in_'):
        print('feature_names_in_len:', len(model.feature_names_in_))
    try:
        print('model_attributes sample:', [a for a in dir(model) if not a.startswith('_')][:50])
    except Exception as exc:
        print('model inspect error:', exc)
