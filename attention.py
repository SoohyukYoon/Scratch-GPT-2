import torch
import torch.nn as nn 

def example(): 
    # Example Embedding Vectors 
    inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your (x^1)
    [0.55, 0.87, 0.66], # journey (x^2)
    [0.57, 0.85, 0.64], # starts (x^3)
    [0.22, 0.58, 0.33], # with (x^4)
    [0.77, 0.25, 0.10], # one (x^5)
    [0.05, 0.80, 0.55]] # step (x^6)
    )

    x_2 = inputs[1] 
    d_in = inputs.shape[1] 
    d_out = 2 

    # Initialize three weights matrices 
    torch.manual_seed(123) # save random seed for reproducibility 
    W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

    # Compute query, key, and value vector 2 
    query_2 = x_2 @ W_query 
    key_2 = x_2 @ W_key 
    value_2 = x_2 @ W_value 
    print(query_2)

    # Compute query, key, and value vectors 
    keys = inputs @ W_key 
    values = inputs @ W_value 
    print("keys.shape: ", keys.shape)
    print("values.shape: ", values.shape)

    # Compute attention score w_22: 
    keys_2 = keys[1] 
    attn_score_22 = query_2.dot(keys_2) 
    print(attn_score_22)

    # Compute attention score w_2i: 
    attn_score_2is = query_2 @ keys.T 
    print(attn_score_2is)

    # Compute attention weights a_2i: 
    d_k = keys.shape[-1] 
    attn_weights_2 = torch.softmax(attn_score_2is / d_k**0.5, dim=-1) # division reduces variance back to 1 
    print(attn_weights_2)

    # Compute context vector_2: 
    context_vector_2 = attn_weights_2 @ values 
    print(context_vector_2)


#### Task: Compute all context vectors #### 
class SelfAttention_v1(nn.Module): 
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
    
    def forward(self, x):
        querys = x @ self.W_query # (n, 2)
        keys = x @ self.W_key # (n, 2)
        values = x @ self.W_value # (n, 2)
        attn_scores = querys @ keys.T # (n, 2) @ (2, n) -> (n, n) 
        d_k = keys.shape[-1] 
        attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1) # (n,n)
        cntxt_vecs = attn_weights @ values # (n, n) @ (n, 2) -> (n, 2)
        return cntxt_vecs

class SelfAttention_v2(nn.Module): 
    def __init__(self, d_in, d_out, qkv_bias=False): # No bias 
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias) 
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    
    def forward(self, x):
        querys = self.W_query(x) # (n, 2)
        keys = self.W_key(x) # (n, 2)
        values = self.W_value(x) # (n, 2)
        attn_scores = querys @ keys.T # (n, 2) @ (2, n) -> (n, n) 
        d_k = keys.shape[-1] 
        attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1) # (n,n)
        cntxt_vecs = attn_weights @ values # (n, n) @ (n, 2) -> (n, 2)
        return cntxt_vecs

class CausalAttention(nn.Module): 
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False): # No bias 
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias) 
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length),
            diagonal=1)
        ) 
    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # Calc attention scores 
        querys = self.W_query(x) # (2, n, 2)
        keys = self.W_key(x) # (2, n, 2)
        values = self.W_value(x) # (2, n, 2)
        attn_scores = querys @ keys.transpose(1,2) # (n, 2) @ (2, 2, n) -> (2, n, n) 

        # Mask: pytorch does it right aligned 
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
       
        # Normalize attenion scores 
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        # Dropout
        attn_weights = self.dropout(attn_weights)
        print(attn_weights)

        # Create Context Vector
        cntxt_vecs = attn_weights @ values # (n, n) @ (n, 2) -> (n, 2)

        return cntxt_vecs

#### Task: MultiHeaded Attention ####
class MultiHeadAttentionWrapper(nn.Module): 
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False): 
        super.__init__()
        self.heads = nn.ModuleList([CausalAttention(d_in, d_out, context_length, dropout)])

    def foward(self, batch): 
        return torch.cat([head(batch) for head in self.heads], dim=-1) # Exercise: Consider if dim=0,1,2 what is the shape?

class MultiHeadAttention(nn.Module): 
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False): 
        super().__init__()
        assert(d_out % num_heads == 0)
        self.d_out = d_out 
        self.num_heads = num_heads 
        self.head_dim = d_out // num_heads 
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_project = nn.Linear(d_out, d_out) 
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
    
    def forward(self, x):
        # Calculate keys, queries, and values | (b, num_tokens, d_out)
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x) 
        queries = self.W_query(x) 
        values = self.W_value(x) 

        # Split into respective heads | (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) # Each token has num_heads each with head_dimn 
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim) 

        # Transpose | (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Calculate Attention Scores | (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3) 
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Calculate Attention Weights | (b, num_heads, num_tokens, num_tokens)
        attn_weights = self.dropout(torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1))
    
        # Calculate Context Vectors | (b, num_tokens, num_heads, head_dim)
        cntxt_vecs = (attn_weights @ values).transpose(1, 2) # (b, num_heads, num_tokens, head_dim) -> (b, num_tokens, num_heads, head_dim)
        cntxt_vecs = cntxt_vecs.contiguous().view(b, num_tokens, self.d_out)

        # Linear Projection
        cntxt_vecs = self.out_project(cntxt_vecs) 
        return cntxt_vecs

if __name__=='__main__': 
    torch.manual_seed(123)
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89], # Your (x^1)
        [0.55, 0.87, 0.66], # journey (x^2)
        [0.57, 0.85, 0.64], # starts (x^3)
        [0.22, 0.58, 0.33], # with (x^4)
        [0.77, 0.25, 0.10], # one (x^5)
        [0.05, 0.80, 0.55]] # step (x^6)
    )
    batch = torch.stack((inputs, inputs), dim=0)
    batch_size, context_length, d_in = batch.shape 
    d_out = 2 
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    cntxt_vecs = mha(batch)
    print(cntxt_vecs) 
    print(cntxt_vecs.shape)
