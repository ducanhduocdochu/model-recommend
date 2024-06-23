from flask import Flask, request, jsonify
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

app = Flask(__name__)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'collaborative_filtering_model.h5')
model = load_model(model_path, compile=False)

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['accuracy'])

# products = pd.read_csv('D:/Code/Hệ hỗ trợ quyết định nhóm 4/api/ratings_Beauty.csv')

num_products = 249274

# Define a function to get the user embeddings
def get_user_embedding(model, user_id):
    user_embedding_model = Model(inputs=model.input[0], outputs=model.get_layer('user_embeddings').output)
    user_embedding = user_embedding_model.predict(np.array([user_id]))
    return user_embedding

# Define a function to get the product embeddings
def get_product_embeddings(model):
    product_embedding_model = Model(inputs=model.input[1], outputs=model.get_layer('product_embeddings').output)
    product_ids = np.arange(num_products)
    product_embeddings = product_embedding_model.predict(product_ids)
    return product_embeddings, product_ids

# Define a function to recommend products for a given user
def recommend_products(model, user_id, top_n=10):
    user_embedding = get_user_embedding(model, user_id)
    product_embeddings, product_ids = get_product_embeddings(model)
    
    # Compute the dot product between the user embedding and all product embeddings
    scores = np.dot(product_embeddings, user_embedding.T).flatten()
    
    # Get the top N products with the highest scores
    top_product_indices = np.argsort(scores)[-top_n:][::-1]
    recommended_products = product_ids[top_product_indices]
    
    return recommended_products.tolist()

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id', type=int)
    
    if user_id is None:
        return jsonify({'error': 'user_id is required'}), 400

    recommended_products = recommend_products(model, user_id, 10)
    
    return jsonify({'recommended_products': recommended_products})

if __name__ == '__main__':
    app.run(debug=True)
