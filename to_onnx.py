import os

import torch

from config.config import config
from test import load_model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    model_weight_path = 'Where Your Model Weight Path'
    onnx_model_save_path = 'Where You Want to Save Your ONNX Model Path'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(
        model_path=model_weight_path,
        device=device
    )
    photometric_size = (config['photo_in_channel'], config['photo_size'], config['photo_size'])
    spectrum_size = (config['spec_in_channel'], config['spectrum_size'])
    # to onnx
    dummy_input = torch.randn(1, *photometric_size).to(device)
    dummy_input2 = torch.randn(1, *spectrum_size).to(device)
    torch.onnx.export(
        model,
        (dummy_input, dummy_input2),
        onnx_model_save_path,
        verbose=False,
        input_names=['photometric', 'spectrum'],
        output_names=['mass_m_sun', 'age_gyr'],
        dynamic_axes={
            'photometric': {0: 'batch_size'},
            'spectrum': {0: 'batch_size'}
        }
    )
