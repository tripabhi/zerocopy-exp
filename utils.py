import torch
import copy
from typing import Dict, List, Tuple


def extract_weights(m: torch.nn.Module) -> Tuple[torch.nn.Module, List[Dict]]:
    tensors = []
    for _, module in m.named_modules():
        params = {
            name: torch.clone(param).detach().numpy()
            for name, param in module.named_parameters(recurse=False)
        }
        buffers = {
            name: torch.clone(buf).detach().numpy()
            for name, buf in module.named_buffers(recurse=False)
        }
        tensors.append({"params": params, "buffers": buffers})

    m_copy = copy.deepcopy(m)
    for _, module in m_copy.named_modules():
        for name in ([name for name, _ in module.named_parameters(recurse=False)]
                     + [name for name, _ in module.named_buffers(recurse=False)]):
            setattr(module, name, None)

    m_copy.train(False)
    return m_copy, tensors


def install_weights(m: torch.nn.Module, tensors: List[Dict]):
    modules = [module for _, module in m.named_modules()]
    for module, tensor_dict in zip(modules, tensors):
        for name, array in tensor_dict["params"].items():
            module.register_parameter(name,
                                      torch.nn.Parameter(torch.as_tensor(array)))
        for name, array in tensor_dict["buffers"].items():
            module.register_buffer(name, torch.as_tensor(array))
