from models import GatedLinearUnit
import numpy as np
import torch

glu = GatedLinearUnit(input_size=5, hidden_layer_size=8, dropout_rate=0)

X = np.arange(0.0, 10.0).reshape(2, 5)
X = torch.tensor(X).double()

glu(X)
