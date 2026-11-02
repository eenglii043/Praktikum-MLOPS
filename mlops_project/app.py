from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.get('/health')
def health():
    return jsonify({'status': 'ok', 'model': 'baseline'})

@app.post('/predict')
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or 'features' not in payload:
        return jsonify({'error': 'Kirim JSON dengan kunci features'}), 400
    values = payload['features']
    if not isinstance(values, list) or len(values) != 4:
        return jsonify({'error': 'features harus berisi tepat empat angka'}), 400
    try:
        features = np.asarray(values, dtype=float).reshape(1, 4)
        if not np.isfinite(features).all():
            raise ValueError('Nilai tidak finite')
    except (ValueError, TypeError):
        return jsonify({'error': 'features harus angka valid'}), 400
    return jsonify({'prediction': model.predict(features).tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
