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
class FackList:
  def __init__(self) -> None:
    self.data = []

# A model parallel
class ModelParallel(nn.Module):
  def __init__(self):
    super(ModelParallel, self).__init__()
    self.tmp_sub_module = FackList()

  # Manage a sub module
  # weight: The weight of a submodule on a device
  def mp_m(self, module, weight = 1):
    self.tmp_sub_module.data.append([module, weight, None])
    return module

  # Apply forward for module
  def mp_f(self, module, x):
    x = x.to(next(module.parameters()).device)
    return module(x)

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
