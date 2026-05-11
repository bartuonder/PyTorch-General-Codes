import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):

    def __init__(self,
                 embedding_dim: int = 384,
                 num_heads: int = 6,
                 dropout: float = 0.1
                 ):
        super().__init__()

        self.ln1 = nn.LayerNorm(embedding_dim)

        self.attention = nn.MultiheadAttention(embed_dim = embedding_dim,
                                               num_heads = num_heads,
                                               dropout = dropout,
                                               batch_first = True)

        self.ln2 = nn.LayerNorm(embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_features = embedding_dim, out_features = embedding_dim * 4),
            nn.GELU(),
            nn.Linear(in_features=embedding_dim * 4, out_features=embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:

        x_norm = self.ln1(x)

        attn_output, _ = self.attention(query = x_norm,
                                        key = x_norm,
                                        value = x_norm,
                                        attn_mask = causal_mask,
                                        is_causal = False
                                        )

        x = x + attn_output

        x = x + self.mlp(self.ln2(x))

        return x



class GPT(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 384,
                 num_heads: int = 6,
                 dropout: float = 0.1,
                 block_size: int = 256,
                 num_layers: int = 6
                 ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embedding_dim
        )

        self.position_embedding = nn.Embedding(
            num_embeddings = block_size,
            embedding_dim = embedding_dim
        )

        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    embedding_dim = embedding_dim,
                    num_heads = num_heads,
                    dropout = dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = nn.LayerNorm(embedding_dim)

        self.output_proj = nn.Linear(in_features = embedding_dim, out_features = vocab_size)

        self.loss_fn = nn.CrossEntropyLoss()

        causal_mask = torch.triu(
            torch.ones(block_size, block_size, dtype = torch.bool),
            diagonal = 1
        )

        self.register_buffer("causal_mask", causal_mask)

        self.apply(self._init_weights)

        total_params = sum(p.numel() for p in self.parameters())
        #print(f"Total Parameters: {total_params}")


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None):

        batch_size, seq_len = input_ids.shape

        device = input_ids.device

        token_emb = self.token_embedding(input_ids)

        positions = torch.arange(seq_len, device = device)

        pos_emb = self.position_embedding(positions)

        x = self.dropout(token_emb + pos_emb)

        mask = self.causal_mask[:seq_len, :seq_len]

        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_final(x)

        logits = self.output_proj(x)
        #print(f"Logits Shape: {logits.shape}")

        loss = None

        if targets is not None:

            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
            targets_flat = targets.reshape(batch_size * seq_len)
            loss = self.loss_fn(logits_flat, targets_flat)

        return logits, loss


    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 0.5):

        for _ in range(max_new_tokens):

            if input_ids.size(1) <= self.block_size:
                current_input = input_ids

            else:
                current_input = input_ids[:, -self.block_size:]

            logits, _ = self.forward(current_input)

            last_logits = logits[:, -1, :]

            last_logits = last_logits / temperature

            probs = F.softmax(last_logits, dim = -1)

            next_token = torch.multinomial(probs, num_samples = 1)

            input_ids = torch.cat([input_ids, next_token], dim = 1)

        return input_ids