import joblib

try:
    model = joblib.load('Delivery_Time.pkl')
    print(f"Object: {type(model)}")
    if hasattr(model, 'predict'):
        print("  Has predict method!")
except Exception as e:
    print(f"Error loading model: {e}")

