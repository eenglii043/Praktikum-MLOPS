# Praktikum MLOps Iris — Colab + Codespaces

Sumber data sama: `load_iris()` dari scikit-learn (150 sampel, empat fitur). Semua model dilatih pada **baris training identik** (120 sampel; `test_size=0.2`, `random_state=42`) dan dites pada 30 sampel identik.

- Baseline: `model.pkl` + `app.py`, JSON `{"features":[5.1,3.5,1.4,0.2]}`.
- Blue: `model_blue.pkl` + `app_blue.py` (Logistic Regression), JSON `{"data":[5.1,3.5,1.4,0.2]}`.
- Green: `model_green.pkl` + `app_green.py` (Random Forest), JSON `{"data":[5.1,3.5,1.4,0.2]}`.

Port host dalam Codespaces: baseline 5002; Blue 5000; Green 5001.

Docker:
```
docker build -t iris:blue -f Dockerfile.blue .
docker build -t iris:green -f Dockerfile.green .
docker run -d --name blue -p 5000:5000 iris:blue
docker run -d --name green -p 5001:5000 iris:green
```

Contoh uji:
```
curl -X POST http://localhost:5000/predict -H 'Content-Type: application/json' -d '{"data":[5.1,3.5,1.4,0.2]}'
```
