from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('model_blue.pkl')
VERSION = 'Blue'
SPECIES = ['setosa', 'versicolor', 'virginica']

def parse_data(payload):
    if not isinstance(payload, dict) or 'data' not in payload:
        raise ValueError('Kirim JSON dengan kunci data')
    values = payload['data']
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError('data harus berisi tepat 4 angka')
    try:
        matrix = np.asarray(values, dtype=float).reshape(1, 4)
    except (TypeError, ValueError) as exc:
        raise ValueError('data harus berupa angka') from exc
    if not np.isfinite(matrix).all():
        raise ValueError('data tidak boleh NaN/Infinity')
    return matrix

@app.get('/health')
def health():
    return jsonify({'status':'ok','version':VERSION})

@app.get('/model-info')
def model_info():
    return jsonify({'version':VERSION,'algorithm':'LogisticRegression',
                    'features':['sepal length','sepal width','petal length','petal width'],
                    'classes':SPECIES})

@app.post('/predict')
def predict():
    try:
        features = parse_data(request.get_json(silent=True))
    except ValueError as exc:
        return jsonify({'error':str(exc)}), 400
    result = model.predict(features).astype(int).tolist()
    return jsonify({'version':VERSION,'prediction':result,'species':[SPECIES[i] for i in result]})

@app.post('/predict-batch')
def predict_batch():
    payload = request.get_json(silent=True)
    if not isinstance(payload,dict) or not isinstance(payload.get('instances'),list) or not 1 <= len(payload['instances']) <= 100:
        return jsonify({'error':'instances harus list dengan 1-100 baris'}), 400
    try:
        matrix = np.vstack([parse_data({'data':row}) for row in payload['instances']])
    except ValueError as exc:
        return jsonify({'error':str(exc)}), 400
    result = model.predict(matrix).astype(int).tolist()
    return jsonify({'version':VERSION,'prediction':result,'species':[SPECIES[i] for i in result]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
