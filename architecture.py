GPT_CONFIG_124M = { 
    "vocab_size": 50257, 
    "context_length": 1024, 
    "emb_dim": 768, 
    "n_heads": 12, 
    "n_layers": 12, 
    "drop_rate": 0.1, 
    "qkv_bias": False # Bias in Linear Layer
}

import torch 
import torch.nn as nn 
from attention import MultiHeadAttention
class GPTModel(nn.Module): 
    def __init__(self, cfg): 
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"]) 
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx): 
        batch_size, seq_len = in_idx.shape 
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        x = tok_embeds + pos_embeds 
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x) 
        logits = self.out_head(x)
        return logits 

class LayerNorm(nn.Module): # sends outputs variance=1, mean=0
    def __init__(self, emb_dim): 
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim)) # (emb_dim, )
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x): 
        mean = x.mean(dim=-1, keepdim=True) # default is n, though mathematically to get rid of bias it should be n - 1
        var = x.var(dim=-1, keepdim=True, unbiased=False) 
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift # (emb_dim,) broadcasted to (1, 1, emb_dim)

class GELU(nn.Module): 
    def __init__(self): 
        super().__init__()
    
    def forward(self, x): 
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
    
class FeedForward(nn.Module): 
    def __init__(self, cfg): 
        super().__init__() 
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
            GELU(), 
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim'])
        )
    
    def forward(self, x):
        '''
        (b, num_tokens, emb_dim)
        -> (b, num_token, emb_dim * 4) # First Linear Output
        -> (b, num_token, emb_dim * 4) # GELU Output
        -> (b, num_token, emb_dim) # Second Linear Output 
        '''
        return self.layers(x)

class TransformerBlock(nn.Module): 
    '''
    Takes in the batch 
    Outputs contexts vectors for each corresponding token, context vector embedding dimension is also 768
    '''
    def __init__(self, cfg): 
        super().__init__()
        
        # "Regularize to prevent overfitting"
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.ln_1 = LayerNorm(cfg['emb_dim'])
        self.ln_2 = LayerNorm(cfg['emb_dim'])

        self.ff = FeedForward(cfg)
        self.gelu = GELU()
        self.mha = MultiHeadAttention(
            d_in=cfg['emb_dim'], 
            d_out=cfg['emb_dim'], 
            context_length=cfg['context_length'], 
            dropout=cfg['drop_rate'], 
            num_heads=cfg['n_heads'], 
            qkv_bias=cfg['qkv_bias']
        )
    
    def forward(self, x): 
        short = x 
        x = self.ln_1(x)
        x = self.mha(x)
        x = short + self.dropout(x)

        short = x 
        x = self.ln_2(x) 
        x = self.ff(x)
        x = short + self.dropout(x)

        return x 

def main(): 
    import tiktoken 
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = [] 
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    
    # Get IDs
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    print(batch)

    # Initialize DummyGPTModel
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    
    start_context = "Hello, I am "
    idx = tokenizer.encode(start_context)
    idx_tensor = torch.tensor(idx).unsqueeze(0) # Adds a batch dimension (1, token, emb_dim)
    
    model.eval()
    out = generate_text_simple(model=model,
                               idx=idx_tensor, 
                               max_new_tokens=5, 
                               context_size=GPT_CONFIG_124M['context_length']
                            )   
    print("Output: ", out)
    print("Output Lenght: ", len(out[0]))

    decoded_out = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_out)

def generate_text_simple(model, idx, max_new_tokens, context_size): 
    for _ in range(max_new_tokens): 
        data = idx[:,-context_size:]
        with torch.no_grad(): # No gradient calculation---more efficient
            logits = model(data)
        logits = logits[:,-1,:]
        probs = torch.softmax(logits, dim=-1)
        new_idx = torch.argmax(probs, dim=-1, keepdim=True)
        idx = torch.cat((idx, new_idx), dim=1)

    return idx 

if __name__=="__main__": 
    main()
    
