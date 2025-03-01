import xgboost as xgb

model = xgb.XGBClassifier()
model.load_model("model.xgb")
