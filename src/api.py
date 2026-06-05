from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os
import uvicorn
from contextlib import asynccontextmanager

# Variável global para armazenar o modelo carregado
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    # Tenta carregar o modelo de caminhos relativos ao arquivo ou à execução
    model_path = os.path.join(os.path.dirname(__file__), '../models/xgboost_clv_model.pkl')
    if not os.path.exists(model_path):
        model_path = 'models/xgboost_clv_model.pkl'
        
    try:
        model = joblib.load(model_path)
        print(f"Modelo carregado com sucesso a partir de: {model_path}")
    except Exception as e:
        print(f"Erro ao carregar o modelo a partir de {model_path}: {e}")
        raise e
    yield
    # Limpeza (opcional)
    pass

app = FastAPI(
    title="CLV Prediction API",
    description="API REST para predição de Customer Lifetime Value (CLV) baseada em métricas RFM e clusterização.",
    version="1.0.0",
    lifespan=lifespan
)

class CustomerData(BaseModel):
    recency: float = Field(..., description="Recência do cliente em dias (R)", example=100.0)
    frequency: float = Field(..., description="Frequência de compras do cliente (F)", example=1.0)
    cluster_id: int = Field(..., description="ID do cluster determinado pelo K-Means", example=0)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "CLV Prediction API está funcionando normalmente.",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict(data: CustomerData):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo preditivo não inicializado ou indisponível.")
    
    try:
        # Converter os dados de entrada em DataFrame do Pandas
        # O XGBoost exige os mesmos nomes de colunas do treinamento para evitar warnings/erros
        input_df = pd.DataFrame([{
            'recency': data.recency,
            'frequency': data.frequency,
            'cluster_id': data.cluster_id
        }])
        
        # Realizar a predição
        prediction = model.predict(input_df)
        predicted_value = float(prediction[0])
        
        return {
            "predicted_clv": round(predicted_value, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro durante a predição: {str(e)}")

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
