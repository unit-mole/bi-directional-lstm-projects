import torch
from src.attention_layer import TemporalAttention
def test_attention_shape_and_normalization():
    layer=TemporalAttention(8); sequence=torch.randn(2,5,8); mask=torch.tensor([[1,1,1,0,0],[1,1,1,1,1]],dtype=torch.bool); context,weights=layer(sequence,mask); assert context.shape==(2,8); assert weights.shape==(2,5); assert torch.allclose(weights.sum(1),torch.ones(2),atol=1e-5); assert weights[0,3:].max().item()<1e-6
