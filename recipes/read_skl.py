import joblib
import numpy as np

with open("/tmp/6adf97f83acf6453d4a6a4b1070f3754.pkl", "rb") as f:
    model = joblib.load(f)

print(model.predict_log_proba(np.array([[1, 2, 3, 4]])))
