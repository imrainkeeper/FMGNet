import numpy as np
import torch
import torch.nn.functional as F


tensor = torch.rand([1, 2, 8, 8])
print(tensor)
print(tensor.shape)

# tensor1 = F.max_pool2d(tensor, kernel_size=2, stride=2)
# print(tensor1)
# print(tensor1.shape)

reshape_tensor = tensor.permute(0, 2, 3, 1)
print(reshape_tensor)
print(reshape_tensor.shape)

reshape_tensor1 = reshape_tensor.reshape([64, 2])
print(reshape_tensor1)
print(reshape_tensor1.shape)

reshape_tensor2 = F.softmax(reshape_tensor1, dim=1)
print(reshape_tensor2)
print(reshape_tensor2.shape)