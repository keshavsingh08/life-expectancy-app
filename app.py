from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# 1. Load your specific uploaded files
model = joblib.load('life_expectancy_model.pkl')
encoder = joblib.load('country_encoder.pkl')

@app.route('/')
def home():
    # We pass the list of countries to the dropdown menu in the HTML
    # derived from your encoder's classes (e.g., Afghanistan, India, etc.)
    countries = sorted(encoder.classes_)
    return render_template('index.html', countries=countries)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 2. Get data from the form
        selected_country = request.form['country']
        selected_year = float(request.form['year'])

        # 3. Preprocess the input
        # The model expects [Entity_Code, Year]
        # We use your encoder to turn "India" or "United States" into a number
        entity_code = encoder.transform([selected_country])[0]
        
        features = np.array([[entity_code, selected_year]])

        # 4. Make Prediction
        prediction = model.predict(features)
        output = round(prediction[0], 2)

        return render_template('index.html', 
                               prediction_text=f'Predicted Life Expectancy for {selected_country} in {int(selected_year)}: {output} years',
                               countries=sorted(encoder.classes_))
                               
    except Exception as e:
        return render_template('index.html', 
                               prediction_text=f'Error: {str(e)}',
                               countries=sorted(encoder.classes_))

if __name__ == "__main__":
    app.run(debug=True, port=5000)