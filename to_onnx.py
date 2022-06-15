import torch
import torch.nn as nn

def to_onnx():
    model_file_path = "./mobilenetv3_small.pkl"
    onnx_file_path = "./mobilenetv3_small.onnx"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = torch.load(model_file_path, map_location = device)
    model = model.to(device)
    # print(model)

    input_img = torch.randn(1, 1, 224, 224, device = device)

    input_names = ["input_data"]
    output_names = ["output_data"]
    
    torch.onnx.export(model, input_img, onnx_file_path, verbose=True, input_names=input_names, output_names=output_names)
    print("onnx completed!")
