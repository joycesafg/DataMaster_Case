
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import uvicorn
import pickle
import os
import xgboost

print(os.listdir("."))
# Carregar o modelo
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

print(model)
# Iniciar a aplicação FastAPI
app = FastAPI()

# Definir a estrutura do payload da solicitação
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Endpoint para previsões
@app.post("/predict")
def predict(iris: IrisRequest):
    data = np.array([[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]])
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}

# if __name__=="__main__":
#     app.run(debug=True)
    #uvicorn.run(app, host="127.0.0.1", port=8001)