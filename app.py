from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from sklearn.naive_bayes import GaussianNB

model = joblib.load("model.pkl")

# request body schema
class InputData(BaseModel):
    gula_darah: float
    tekanan_darah: float

app = FastAPI()

@app.get("/")
async def root():
    """Endpoint utama untuk memeriksa apakah API berjalan."""
    return {"message": "Selamat Datang di API Prediksi Diabetes. Gunakan endpoint /docs untuk melihat dokumentasi."}

@app.post("/predict")
def predict(data: InputData):
    X = [[data.gula_darah, data.tekanan_darah]]
    prediction = model.predict(X)[0]
    return {"prediction": int(prediction)}
