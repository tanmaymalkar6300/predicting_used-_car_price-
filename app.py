from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load your trained ML model (make sure the model file is correctly located)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract form data
            brand = request.form['brand']
            model_name = request.form['model']
            price = float(request.form['price'])
            body = request.form['body']
            mileage = float(request.form['mileage'])
            engine_v = float(request.form['engine_v'])
            engine_type = request.form['engine_type']
            registration = 1 if request.form['registration'] == 'yes' else 0
            year = int(request.form['year'])

            # Preprocessing: Ensure your model expects this format
            input_features = [price, mileage, engine_v, registration, year]

            # Predict using the trained model
            prediction = model.predict([input_features])
            output = round(prediction[0], 2)

            # Render the result back to the template
            return render_template('index.html', output=f"{output} USD")

        except Exception as e:
            return render_template('index.html', output=f"Error: {str(e)}")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True,port=5000)
