import torch
from thop import profile
import copy

def calculate_flops(model, input_size=(3, 224, 224)):
    dummy_input = torch.randn(1, *input_size).to(next(model.parameters()).device)
    flops, params = profile(copy.deepcopy(model), inputs=(dummy_input,), verbose=False)
    
    if flops > 1e9:
        flops_str = f"{flops / 1e9:.2f} GFLOPs"
    elif flops > 1e6:
        flops_str = f"{flops / 1e6:.2f} MFLOPs"
    else:
        flops_str = f"{flops / 1e3:.2f} KFLOPs"
        
    if params > 1e6:
        params_str = f"{params / 1e6:.2f}M params"
    elif params > 1e3:
        params_str = f"{params / 1e3:.2f}K params"
    else:
        params_str = f"{params} params"
        
    return flops, params, flops_str, params_str