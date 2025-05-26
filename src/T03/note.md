# PyTorch Model Experimentation Notes

## Experimenting with MLP (Multilayer Perceptron) Layers Only

`mlp_attn_min.ipynb`

- Feature Usage:
  - Attempting to use only 3 features from the tabular data does not result in successful training or good performance.
  - All 47 features are required for the MLP to effectively learn and deliver reasonable results.
- ReLU Layer Placement:
  - Placing a ReLU activation function in the early layers of the MLP seems to cause issues.
  - It is more effective and stable to apply the ReLU activation at the final MLP layers rather than at the beginning.

## Experimenting with MLP + LSTM (Encoder-Decoder) + Attention

`mlp_attn.ipynb`

- Performance with Standard Setup:
  - Using a combination of LSTM encoder-decoder with an attention network (with all 47 features) does not yield good performance. The network fails to learn or provide improved results over the MLP-only approach.
- Residual Connection as a Workaround:
  - Adding a residual connection (skipping LSTM and Attention layers) is the only method found so far to achieve reasonable results.
  - However, this approach renders the LSTM and Attention layers unused—the model essentially bypasses them.
  - Inspecting attention values confirms their low magnitude, indicating that the network is not relying on them for learning or prediction.

Summary:

For MLPs, ensure to use the full set of features and be cautious with where ReLU activations are introduced.
The current LSTM + Attention integration does not improve performance unless essentially skipped—which also makes those layers ineffective in practice.
