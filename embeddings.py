import re
class SimpleTokenizerV2: 

    def __init__(self, vocab): 
        self.str_to_int = vocab 
        self.int_to_str = {integer:token for token, integer in vocab.items()}
    
    def encode(self, raw_text): 
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[token] for token in preprocessed]
        return ids 
        
    def decode(self, ids): 
        text = ' '.join(self.int_to_str[i] for i in ids)
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text 

def simple_tokenizer(raw_text): 
    #### CREATING VOCABULARY #### 
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]

    all_words = sorted(set(preprocessed))
    all_words.extend(["<|endoftext|>", "<|unk|>"])
    vocab_size = len(all_words)
    print(vocab_size)

    vocab = {token:integer for integer, token in enumerate(all_words)}
    tokenizer = SimpleTokenizerV2(vocab)
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))
    print(text)
    ids = tokenizer.encode(text)
    print(ids)
    print(tokenizer.decode(ids))

import torch 
from torch.utils.data import DataLoader, Dataset 
from importlib.metadata import version
import tiktoken

class GPTDatasetV1(Dataset): 
    def __init__(self, txt, tokenizer, max_length, stride): 
        self.input_ids = [] 
        self.target_ids = [] 

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride): 
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + 1 + max_length]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self): 
        return len(self.input_ids)
    
    def __getitem__(self, idx): 
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

def generate_embeddings(raw_text): 
    '''
    For word embeddings, for each word instead of immediately assigning a token it begins from individuals characters. Continually they merge the character into subwords, and merges subwords together. The decision for upto when and which characters they merge is "outside the scope of this book" --- maybe something extra I can do. But key idea is merging based on which merge is most likely, frequency. 
    '''
    # Initiliaze word embedding layers 
    vocab_size = 50257 # bpe has vocab 5027 ids 
    output_dim = 256 # min dim for making llms 
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim) 

    # Initialize positional embedding layers 
    context_length = max_length = 4
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
   
    input_embeddings = pos_embeddings + token_embedding_layer
    print(input_embeddings.shape())

    # Create the dataset/loader 
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=1, shuffle=False)
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    first_input, first_target = first_batch 

    # Ouput initialized embeddings from first batch 
    print(token_embedding_layer(first_input), pos_embeddings(first_input)) # prints, torch.Size([8,4,256])

if __name__=='__main__': 
    #### DOWNLOAD STORY ####
    import ssl
    import urllib.request

    url = ("https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
    "the-verdict.txt")
    file_path = "the-verdict.txt"

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    with urllib.request.urlopen(url, context=ctx) as resp:
        with open(file_path, "wb") as f:
            f.write(resp.read())

    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    # simple_tokenizer(raw_text)
    generate_embeddings(raw_text)