import joblib
bundle = joblib.load('models/artifacts/cnn_bundle.joblib')
print(bundle.keys())
