from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model
model = load_model('collaborative_filtering_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_id = data['UserId']
    product_id = data['ProductId']
    
    # Reshape the inputs to match the model's expected input shape
    user_input = np.array([user_id])
    product_input = np.array([product_id])
    
    # Make the prediction
    prediction = model.predict([user_input, product_input])
    
    # Return the result as a JSON response
    return jsonify({'prediction': float(prediction[0][0])})

if __name__ == '__main__':
    app.run(debug=True)
