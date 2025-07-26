from flask import Flask, render_template, request
import pandas as pd
import json
import pickle


app = Flask(__name__)

# Load model once at startup
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("preprocessor.pkl", "rb") as f1:
    preprocessor = pickle.load(f1)

# Load dropdown options once at startup
with open("cities.json") as f:
    cities = json.load(f)
with open("streets.json") as f:
    streets = json.load(f)
with open("state_zips.json") as f:
    state_zips = json.load(f)

# Columns
categorical_cols = ['city', 'street', 'statezip']
numerical_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                  'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
                  'yr_built', 'yr_renovated']

@app.route("/", methods=["GET", "POST"])
def info():
    predicted_price = None  # Default

    if request.method == "POST":
        # Get form data
        city = request.form.get("city")
        street = request.form.get("street")
        state_zip = request.form.get("state_zip")
        sqft_living = float(request.form.get("sqft_living"))
        sqft_lot = float(request.form.get("sqft_lot"))
        bedrooms = float(request.form.get("bedrooms"))
        bathrooms = float(request.form.get("bathrooms"))
        floors = float(request.form.get("floors"))
        waterfront = float(request.form.get("waterfront"))
        view = float(request.form.get("view"))
        condition = float(request.form.get("condition"))
        sqft_basement = float(request.form.get("sqft_basement"))
        sqft_above = float(request.form.get("sqft_above"))
        yr_built = int(request.form.get("yr_built"))
        yr_renovated = int(request.form.get("yr_renovated"))

        # Create DataFrame
        df_input = pd.DataFrame([{
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'sqft_living': sqft_living,
            'sqft_lot': sqft_lot,
            'floors': floors,
            'waterfront': waterfront,
            'view': view,
            'condition': condition,
            'sqft_above': sqft_above,
            'sqft_basement': sqft_basement,
            'yr_built': yr_built,
            'yr_renovated': yr_renovated,
            'street': street,
            'city': city,
            'statezip': state_zip
        }])

        # Predict price
        predicted_price = predict_price(df_input)
    return render_template("index.html", cities=cities, streets=streets, state_zips=state_zips, predicted_price=predicted_price)


# Encodes and scales input, then predicts
def predict_price(X):
    # categorical_cols = ['city', 'street', 'statezip']
    # numerical_cols = [col for col in X.columns if col not in categorical_cols]

    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols),
    #         ('num', StandardScaler(), numerical_cols)
    #     ]
    # )

    X_processed = preprocessor.transform(X)
    print(f"Processed input: {X_processed}")
    price = model.predict(X_processed)  # Return single prediction
    print(f"Predicted price: {price[0]}")
    return price[0]

if __name__ == "__main__":
    app.run(debug=True)
