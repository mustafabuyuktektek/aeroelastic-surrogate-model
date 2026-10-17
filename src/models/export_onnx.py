import torch
import os
from train_dnn import SurrogateDNN

def export_model():
    model = SurrogateDNN()
    model_path = os.path.join(os.path.dirname(__file__), 'dnn_model.pth')
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    dummy_input = torch.randn(1, 4)
    # API klasörüne kaydet
    api_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../api'))
    os.makedirs(api_dir, exist_ok=True)
    onnx_path = os.path.join(api_dir, 'aero_surrogate.onnx')

    torch.onnx.export(model, dummy_input, onnx_path,
                      input_names=['input_params'], output_names=['flutter_load'])
    print(f"Model ONNX formatında API klasörüne aktarıldı: {onnx_path}")

if __name__ == '__main__':
    export_model()