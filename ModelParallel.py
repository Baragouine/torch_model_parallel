import types

import numpy as np

import torch
import torch.nn as nn


# Get free gpu memory
def get_device_free_memory(dev):
  total_memory = torch.cuda.get_device_properties(dev).total_memory
  allocated_memory = torch.cuda.memory_allocated(dev)
  return total_memory - allocated_memory


# Ignore: not important
class FakeList:
  def __init__(self) -> None:
    self.data = []

# A model parallel
class ModelParallel(nn.Module):
  def __init__(self):
    super(ModelParallel, self).__init__()
    self.tmp_sub_module = FakeList()

  # Manage a layer
  # weight: The weight of a submodule on a device
  def mp_l(self, module_or_parameter):
    weight = 0
    if isinstance(module_or_parameter, nn.Module):
      weight = sum(p.numel() for p in module_or_parameter.parameters())
    if isinstance(module_or_parameter, nn.Parameter):
      weight = module_or_parameter.numel()
    self.tmp_sub_module.data.append([module_or_parameter, weight, None])
    return module_or_parameter

  # Apply forward for module
  def mp_f(self, module, x):
    x = x.to(next(module.parameters()).device)
    return module(x)

  # Return the device of the parameter
  def mp_device(self, module_or_parameter):
    if isinstance(module_or_parameter, ModelParallel):
      return torch.device('')
    if isinstance(module_or_parameter, nn.Module):
      return next(module_or_parameter.parameters()).device
    if isinstance(module_or_parameter, nn.Parameter):
      return module_or_parameter.device
    if hasattr(module_or_parameter, 'device'):
      return module_or_parameter.device
    raise AttributeError("device is not defined on input object or input object is not a module or a parameter.")

  # send sub module to devices
  def to_devices(self, devices):
    gpus_memory = [get_device_free_memory(dev) for dev in devices]
    total_gpus_memory = sum(gpus_memory)
    sum_weights = sum([e[1] for e in self.tmp_sub_module.data])

    max_weights_gpu = [gpus_memory[i] * sum_weights / total_gpus_memory for i in range(len(gpus_memory))]
    weights_counter = [0 for _ in devices]

    indices_gpu = np.argsort(np.array(gpus_memory))[::-1]

    j = 0
    for i in indices_gpu:
      while weights_counter[i] < max_weights_gpu[i]:
        if j < len(self.tmp_sub_module.data):
          self.tmp_sub_module.data[j][2] = devices[i]
          self.tmp_sub_module.data[j][0].to(devices[i])
          weights_counter[i] += self.tmp_sub_module.data[j][1]
          j += 1
        else:
          break
