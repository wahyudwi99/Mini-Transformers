import torch.nn as nn
from modules.models import MultiHeadAttention, FeedForward
from modules.models import TokenEmbedding, PositionalEncoding


class TransformersBlock(nn.Module):
    """
    Assemblying every components (attention mechanism and FFN).

    Result:
        Transformers Block architecture
    """
    def __init__(self, d_model, num_heads, ffn_hidden, vocab_size):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, ffn_hidden)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)


    def forward(self, x):
        attn_out = self.attn(x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x
    
class MiniTransformers(nn.Module):
    def __init__(self, d_model, num_heads, ffn_hidden, vocab_size, total_blocks):
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.stacked_transformers = nn.ModuleList(
            TransformersBlock(
                d_model,
                num_heads,
                ffn_hidden,
                vocab_size
            ) for _ in range(total_blocks)
        )

        self.llm_head = nn.Linear(d_model, vocab_size)


    def forward(self, x):
        # Embedding
        x = self.embedding(x)

        # Transformers Block
        for layer in self.stacked_transformers:
            x = layer(x)

        # LLM Head (Final Layer)
        x = self.llm_head(x)

        return x