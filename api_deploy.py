from flask import Flask, request, jsonify
from inference_facebook_model import Infere

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_query = data['query']
    mapped_result = Infere.predict(user_query)
    return jsonify(mapped_result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

    # curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"query": "Send $50 to John"}'