import numpy as np
import torch
import torch.nn.functional as F

tensor = torch.rand([2, 3, 3])
print(tensor)
print(tensor.shape)

tensor1 = tensor.view(2, -1, 1)
print(tensor1)
print(tensor1.shape)

tensor2 = tensor1.view(2, -1, 3)
print(tensor2)
print(tensor2.shape)

tensor3 = tensor2.permute(0, 2, 1)
print(tensor3)
print(tensor3.shape)

tensor4 = F.softmax(tensor3, dim=2)
print(tensor4)