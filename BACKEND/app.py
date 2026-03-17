from flask import Flask, request, jsonify
from data import generate_data
from model import ToxicityModel

app = Flask(__name__)

# Initialize
model = ToxicityModel()
df = generate_data()
model.train(df)

@app.route('/')
def home():
    return "AI Toxicity Prediction API Running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    prob, pred = model.predict(data)
    explanation = model.explain()

    return jsonify({
        "toxicity_risk": float(prob),
        "prediction": int(pred),
        "top_factors": explanation
    })

if __name__ == '__main__':
    app.run(debug=True)