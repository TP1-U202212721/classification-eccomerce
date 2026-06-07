from flask import Flask, request, jsonify
from validation.test import predict_risk
app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
   try:
        data = request.get_json()
        result = predict_risk(data)
        if hasattr(result, "item"):
            result = result.item()
        return jsonify({'result': result})
   except Exception as e:
        return jsonify({'error': str(e)}), 400

app.run(port=6000)  