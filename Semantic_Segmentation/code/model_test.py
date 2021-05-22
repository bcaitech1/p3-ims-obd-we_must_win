import torch

from builder.models import SwinTransformerBase

model = SwinTransformerBase().cuda()
print(model)

tensor_in = torch.randn((2, 3, 512, 512)).cuda()
tensor_out = model(tensor_in)
print(tensor_out.shape)
