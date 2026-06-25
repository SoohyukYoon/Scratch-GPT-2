from architecture import * 
from embeddings import * 
from attention import * 

def train_model_simple(model, train_loader, val_loader, 
                       optimizer, device, num_epochs, 
                       eval_freq, eval_iter, start_context, tokenizer): 
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs): 
        model.train() # Inside, since per epoch eval func may set mode.eval()
        for input_batch, target_batch in train_loader: 
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # calc gradients 
            optimizer.step() # update weights 

            # misc performance tracking  
            tokens_seen += input_batch.numel()
            global_step += 1 
            if global_step % eval_freq == 0: 
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}")
        generate_and_print_sample(model, tokenizer, device, start_context, temperature=1.4, top_k=25)
    
    return train_losses, val_losses, track_tokens_seen # tokens seen useful for plotting

# Eval Helpers 
def evaluate_model(model, train_loader, val_loader, device, eval_iter): 
    model.eval()

    with torch.no_grad(): 
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    
    model.train()
    return train_loss, val_loss 

def generate_and_print_sample(model, tokenizer, device, start_context, temperature=1.0, top_k=None):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad(): 
        max_new_tokens = 50
        token_ids = generate(model, encoded, max_new_tokens, context_size, temperature, top_k)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " ")) # compact print format: replace new line with space
    model.train()

import matplotlib.pyplot as plt 
from matplotlib.ticker import MaxNLocator 
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()

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

    # Tokenizer Ready 
    import tiktoken 
    tokenizer = tiktoken.get_encoding("gpt2")

    # Model Ready 
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=0.0004, weight_decay=0.1 # study weight decay 
    )
    num_epochs = 20
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device, 
        num_epochs=num_epochs, eval_freq=5, eval_iter=5, 
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

if __name__=='__main__': 
    main()