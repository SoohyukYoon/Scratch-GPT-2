GPT_CONFIG_124M = { 
    "vocab_size": 50257, 
    "context_length": 256, 
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

# Helpers
def generate(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None, eos_id=None): 
    for _ in range(max_new_tokens): 
        data = idx[:,-context_size:]
        with torch.no_grad(): # No gradient calculation---more efficient
            logits = model(data)
        logits = logits[:,-1,:]
        if top_k is not None:
            top_logits, top_pos = torch.top_k(logits, top_k)
            updated_logits = torch.where(
                condition=new_idx < top_logits[-1], 
                input=float('-inf'), 
                other=logits
            )
        assert temperature > 0., "Temperature value is negative"
        probs = torch.softmax(logits / temperature, dim=-1)
        new_idx = torch.multinomial(probs, num_samples=1)
        if new_idx.item() == eos_id:
            break
        idx = torch.cat((idx, new_idx), dim=1)

    return idx 

def text_to_token_ids(text, tokenizer): 
    token_ids = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    token_ids = torch.tensor(token_ids).unsqueeze(0) 
    return token_ids

def token_ids_to_text(token_ids, tokenizer): 
    flat = token_ids.squeeze(0)
    text = tokenizer.decode(flat.tolist())
    return text 

def calc_loss_batch(input_batch, target_batch, model, device): 
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    # Input dimension: ([batch, seq_len, vocab_size], [batch, seq_len]) 
    # Operation: At seq i, fetch target token prob via logits[b, seq, target_batc[seq]]
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
    return loss 

def calc_loss_loader(data_loader, model, device, num_batches=None): 
    '''
    Function is yo *log* loss of our dataset, not actually use for backprop, since we are finding the average loss of total dataet. 
    Critique: if num_batches < len(data_loader) for efficiency in eval, current function always ignores data_loader[num_batches:] always, not representative, so should shuffle instead in real practice. 
    '''
    total_loss = 0. 
    if len(data_loader) == 0: 
        return float("nan")
    elif num_batches is None: 
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader): 
        if i < num_batches: 
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else: 
            break 
    
    return total_loss / num_batches 
    
# Main
def main():
    # Data Ready 
    from embeddings import create_dataloader_v1
    with open("the-verdict.txt", "r", encoding='utf-8') as file:
        text_data = file.read()

    training_ratio = 0.9
    split = int(training_ratio * len(text_data))
    train_data = text_data[:split]
    val_data = text_data[split:]

    train_loader = create_dataloader_v1(txt=train_data, batch_size=2, max_length=GPT_CONFIG_124M["context_length"], stride=GPT_CONFIG_124M["context_length"], drop_last=True, shuffle=True, num_workers=0)
    val_loader = create_dataloader_v1(txt=val_data, batch_size=2, max_length=GPT_CONFIG_124M["context_length"], stride=GPT_CONFIG_124M["context_length"], drop_last=False, shuffle=False, num_workers=0)

    print("Train loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)
    print("\nValidation loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)

    # Model Ready 
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with torch.no_grad(): 
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)
    
    print("Training loss: ", train_loss)
    print("Validation loss: ", val_loss)

def main_old(): 
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
  
    model.eval()
    out = generate(model=model,
                               idx=text_to_token_ids(start_context, tokenizer), 
                               max_new_tokens=5, 
                               context_size=GPT_CONFIG_124M['context_length']
                            )   
    print("Output: ", out)
    print("Output Lenght: ", len(out[0]))

    print(token_ids_to_text(out, tokenizer))

if __name__=="__main__": 
    main()
    
