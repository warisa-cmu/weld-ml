import torch
from models import VariableSelectionNetwork
import torch.nn as nn

# batch_size = 32
# num_inputs = 1
# input_dim = 8
# hidden_dim = 16
# context_dim = 4

batch_size = 32
hidden_layer_size = 16  # embedding size
dropout_rate = 0.1
output_size = 8  # number of features
input_size = None  # same as hidden_layer_size
additional_context = 5


vsn = VariableSelectionNetwork(
    hidden_layer_size=hidden_layer_size,
    dropout_rate=dropout_rate,
    output_size=output_size,
    input_size=input_size,
    additional_context=additional_context,
)
time = 4
x = torch.randn(batch_size, time, output_size, hidden_layer_size)
context = torch.randn(batch_size, additional_context, hidden_layer_size)
output, weights = vsn((x, context))
print(output.shape)  # (batch_size, hidden_dim)
print(weights.shape)  # (batch_size, num_inputs)
