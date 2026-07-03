import joblib
from pathlib import Path

p = Path('model.pkl')
print('model_exists', p.exists())
if not p.exists():
    raise SystemExit(1)
model = joblib.load(p)
print('type', type(model))
try:
    params = model.get_params()
    print('get_params keys:', list(params.keys())[:50])
except Exception as exc:
    print('get_params error', exc)
for attr in ['feature_names_in_', 'n_features_in_', 'classes_', 'named_steps', 'get_booster', 'booster']:
    print(attr, hasattr(model, attr))
    if hasattr(model, attr):
        try:
            value = getattr(model, attr)
            if callable(value):
                value = value()
            print(attr, type(value))
            if hasattr(value, '__len__') and not isinstance(value, (int, float, str, bytes)):
                print('len', len(value))
                if len(value) <= 100:
                    print(value)
                else:
                    print('first 20', list(value)[:20])
        except Exception as exc:
            print(attr, 'error', exc)
