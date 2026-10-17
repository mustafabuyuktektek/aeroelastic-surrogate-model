from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import os

app = FastAPI(title="Aeroelastic Surrogate Model API")

# ONNX yolunu belirle
onnx_path = os.path.join(os.path.dirname(__file__), 'aero_surrogate.onnx')
ort_session = ort.InferenceSession(onnx_path)

class AeroParams(BaseModel):
    mach: float
    aoa: float
    dynamic_pressure: float
    stiffness: float

@app.post("/predict")
def predict(params: AeroParams):
    input_data = np.array([[params.mach, params.aoa, params.dynamic_pressure, params.stiffness]], dtype=np.float32)
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    ort_outs = ort_session.run(None, ort_inputs)
    return {"flutter_load": float(ort_outs[0][0][0])}