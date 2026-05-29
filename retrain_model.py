import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

print("Loading dataset...")
df = pd.read_csv('Food_Delivery_Times.csv')

# Features and target
features = ['Distance_km', 'Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type', 'Preparation_Time_min', 'Courier_Experience_yrs']
target = 'Delivery_Time_min'

X = df[features]
y = df[target]

print("Defining pipeline...")
numeric_features = ['Distance_km', 'Courier_Experience_yrs', 'Preparation_Time_min']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Match exact model params
model = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('model', RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42))
])

print("Training model...")
model.fit(X, y)

print("Saving model to api/Delivery_Time.pkl...")
joblib.dump(model, 'api/Delivery_Time.pkl')

print("Done! Modern model saved.")
