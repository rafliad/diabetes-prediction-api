from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from sklearn.naive_bayes import GaussianNB
import os

# Dapatkan root_path dari environment variable, defaultnya adalah string kosong ""
root_path = os.environ.get("ROOT_PATH", "")

# load model
model = joblib.load("model.pkl")

# request body schema
class InputData(BaseModel):
    gula_darah: float
    tekanan_darah: float

app = FastAPI(
    title="API Prediksi Diabetes (dari Joblib)",
    root_path=root_path
)

@app.get("/")
async def root():
    """Endpoint utama untuk memeriksa apakah API berjalan."""
    return {"message": "Selamat Datang di API Prediksi Diabetes. Gunakan endpoint /docs untuk melihat dokumentasi."}

@app.post("/predict")
def predict(data: InputData):
    X = [[data.gula_darah, data.tekanan_darah]]
    prediction = model.predict(X)[0]
    return {"prediction": int(prediction)}
