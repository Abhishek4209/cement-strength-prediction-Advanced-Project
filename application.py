import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)
model_path = 'Model/model.pkl'

# Load the dataset and split into features and target:
df = pd.read_csv('D:\CEMENT STREGTH PREDECTION\\notebook\data\\cement_data.csv')
X = df.drop(columns="Concrete compressive strength(MPa, megapascals) ", axis=1)
Y = df["Concrete compressive strength(MPa, megapascals) "]

# Train the model
model = RandomForestRegressor()
model.fit(X, Y)

# Save the trained model
joblib.dump(model, model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_name = ['Cement (component 1)(kg in a m^3 mixture)',
       'Blast Furnace Slag (component 2)(kg in a m^3 mixture)',
       'Fly Ash (component 3)(kg in a m^3 mixture)',
       'Water  (component 4)(kg in a m^3 mixture)',
       'Superplasticizer (component 5)(kg in a m^3 mixture)',
       'Coarse Aggregate  (component 6)(kg in a m^3 mixture)',
       'Fine Aggregate (component 7)(kg in a m^3 mixture)', 
       'Age (day)']

    df = pd.DataFrame([input_features], columns=features_name)
    prediction = model.predict(df)[0]

    return render_template('index.html', prediction_text='Predicted Cement Strength: {:.2f} MPa'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
