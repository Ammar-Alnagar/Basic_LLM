import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    vocab_size: int = 32000
    multiple_of: int = 256
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(batch, slen, n_kv_heads, n_rep, head_dim)
        .reshape(batch, slen, n_kv_heads * n_rep, head_dim)
    )

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * self._norm(x.float()).type_as(x)

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        keys = xk
        values = xv

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = self.dropout(scores)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.dropout(self.wo(output))

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    def forward(self, tokens: torch.Tensor, start_pos: int) -> torch.Tensor:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h)
        return output

class LLaMATokenizer:
    def __init__(self, model_path: str):
        with open(model_path, 'r', encoding='utf-8') as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}
        
    def encode(self, text: str) -> List[int]:
        tokens = []
        for word in text.split():
            if word in self.encoder:
                tokens.append(self.encoder[word])
            else:
                tokens.extend([self.encoder.get(char, self.encoder['<unk>']) for char in word])
        return tokens

    def decode(self, tokens: List[int]) -> str:
        return ' '.join([self.decoder.get(token, '<unk>') for token in tokens])

class LLaMADataset(Dataset):
    def __init__(self, data: List[str], tokenizer: LLaMATokenizer, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = self.tokenizer.encode(text)[:self.max_length]
        return torch.tensor(tokens, dtype=torch.long)

def generate(model: Transformer, tokenizer: LLaMATokenizer, prompt: str, max_new_tokens: int, temperature: float = 0.8, top_p: float = 0.95):
    model.eval()
    tokens = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(next(model.parameters()).device)
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(tokens, start_pos=0)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            
            # Top-p (nucleus) sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            probs[indices_to_remove] = 0
            
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)
    
    generated_text = tokenizer.decode(tokens[0].tolist())
    return generated_text

def train(model: Transformer, tokenizer: LLaMATokenizer, train_data: List[str], val_data: List[str], args: Dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = LLaMADataset(train_data, tokenizer, args['max_seq_len'])
    val_dataset = LLaMADataset(val_data, tokenizer, args['max_seq_len'])
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False)

    optimizer = AdamW(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=args['num_epochs'] * len(train_loader))

    for epoch in range(args['num_epochs']):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args['num_epochs']}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch[:, :-1], start_pos=0)
            loss = F.cross_entropy(logits.view(-1, model.vocab_size), batch[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{args['num_epochs']}, Average Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = batch.to(device)
                logits = model(batch[:, :-1], start_pos=0)
                loss = F.cross_entropy(logits.view(-1, model.vocab_size), batch[:, 1:].reshape(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Generate sample text
        prompt = "Once upon a time"
        generated_text = generate(model, tokenizer, prompt, max_new_tokens=50)
        print(f"Generated Text: {generated_text}")

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
        }, f"checkpoint_epoch_{epoch+1}.pth")

def evaluate_perplexity(model: Transformer, tokenizer: LLaMATokenizer, data: List[str], args: Dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = LLaMADataset(data, tokenizer, args['max_seq_len'])
    dataloader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=False)

    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Perplexity"):
            batch = batch.to(device)
            logits = model(batch[:, :-1], start_pos=0)
            loss = F.cross_entropy(logits.view(-1, model.vocab_size), batch[:, 1:].reshape(-1), reduction='sum')
            total_loss += loss.item()
            total_tokens += batch[:, 1:].numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity

def load_checkpoint(model: Transformer, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, scheduler, epoch, loss

def main():
    # Model hyperparameters
    model_args = ModelArgs(
        dim=2048,
        n_layers=24,
        n_heads=16,
        vocab_size=32000,
        multiple_of=256,
        norm_eps=1e-5,
        max_seq_len=2048,
        dropout=0.1,
    )

    # Training hyperparameters
    train_args = {
        'batch_size': 32,
        'learning_rate': 3e-4,
        'weight_decay': 0.1,
        'num_epochs': 10,
        'max_grad_norm': 1.0,
        'max_seq_len': model_args.max_seq_len,
    }

    # Initialize model, tokenizer, and optimizer
    model = Transformer(model_args)
    tokenizer = LLaMATokenizer("path_to_tokenizer.json")
    optimizer = AdamW(model.parameters(), lr=train_args['learning_rate'], weight_decay=train_args['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=train_args['num_epochs'] * (len(train_data) // train_args['batch_size']))

    # Load data (replace with your actual data loading logic)
    train_data = ["Your training data here"]
    val_data = ["Your validation data here"]
    test_data = ["Your test data here"]

    # Train the model
    train(model, tokenizer, train_data, val_data, train_args)

    # Evaluate perplexity on test set
    test_perplexity = evaluate_perplexity(model, tokenizer, test_data, train_args)
    print(f"Test Perplexity: {test_perplexity:.2f}")

    # Generate some text
    prompt = "In a world where technology"
    generated_text = generate(model, tokenizer, prompt, max_new_tokens=100)
    print(f"Generated Text:\n{generated_text}")

if __name__ == "__main__":
    main()
