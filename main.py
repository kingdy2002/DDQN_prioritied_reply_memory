from torch.distributions import Categorical
import torch
a = torch.tensor([0.5,0.7,0.3])
a= Categorical(a)
print(a)
a = a.sample()
print(a)