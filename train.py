import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import re

def word_tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text)
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR


# --------------------------------------------------
# Configuration
# --------------------------------------------------

batch_size = 32
block_size = 64
max_iters = 1000
eval_interval = 100
learning_rate = 1e-3
eval_iters = 200

n_embd = 64
n_layer = 4
dropout_rate = 0.15

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(1337)


# --------------------------------------------------
# Load text
# --------------------------------------------------

with open("sentences.txt", "r", encoding="utf-8") as f:
    text = f.read()

if not text.strip():
    raise ValueError("sentences.txt is empty.")


# --------------------------------------------------
# Tokenization
# --------------------------------------------------

words = word_tokenize(text)

vocab = sorted(set(words))
vocab_size = len(vocab)

stoi = {word: i for i, word in enumerate(vocab)}
itos = {i: word for i, word in enumerate(vocab)}


def encode(sentence):
    return [stoi[word] for word in word_tokenize(sentence)]


def decode(tokens):
    return " ".join(itos[token] for token in tokens)


# --------------------------------------------------
# Train / validation split
# --------------------------------------------------

data = torch.tensor(
    encode(text),
    dtype=torch.long
)

n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]


if len(train_data) <= block_size or len(val_data) <= block_size:
    raise ValueError(
        "sentences.txt is too small for the selected block_size."
    )


# --------------------------------------------------
# Batch generation
# --------------------------------------------------

def get_batch(split):
    source = train_data if split == "train" else val_data

    ix = torch.randint(
        0,
        len(source) - block_size,
        (batch_size,)
    )

    x = torch.stack(
        [source[i:i + block_size] for i in ix]
    )

    y = torch.stack(
        [source[i + 1:i + block_size + 1] for i in ix]
    )

    x = x.to(device)
    y = y.to(device)

    return x, y


# --------------------------------------------------
# LSTM Language Model
# --------------------------------------------------

class LSTMModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        n_embd,
        n_layer,
        dropout_rate=0.15
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            n_embd
        )

        self.lstm = nn.LSTM(
            input_size=n_embd,
            hidden_size=n_embd,
            num_layers=n_layer,
            batch_first=True,
            dropout=dropout_rate if n_layer > 1 else 0.0,
            bidirectional=False
        )

        self.dropout = nn.Dropout(dropout_rate)

        self.lm_head = nn.Linear(
            n_embd,
            vocab_size
        )

        self._initialize_weights()


    def _initialize_weights(self):

        for name, param in self.lstm.named_parameters():

            if "bias" in name:
                nn.init.constant_(param, 0.0)

            elif "weight" in name:
                nn.init.xavier_normal_(param)

        nn.init.xavier_normal_(self.lm_head.weight)
        nn.init.zeros_(self.lm_head.bias)


    def forward(self, idx, targets=None):

        # idx shape:
        # (batch_size, sequence_length)

        token_embeddings = self.token_embedding(idx)

        # Shape:
        # (batch_size, sequence_length, embedding_size)

        x, hidden = self.lstm(token_embeddings)

        x = self.dropout(x)

        logits = self.lm_head(x)

        # Shape:
        # (batch_size, sequence_length, vocab_size)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            logits_flat = logits.reshape(B * T, C)
            targets_flat = targets.reshape(B * T)

            loss = F.cross_entropy(
                logits_flat,
                targets_flat
            )

        return logits, loss, hidden


    @torch.no_grad()
    def generate(self, idx, max_new_tokens):

        self.eval()

        for _ in range(max_new_tokens):

            # Only use the current context.
            context = idx[:, -block_size:]

            logits, _, _ = self(context)

            # Take prediction for final token.
            logits = logits[:, -1, :]

            probabilities = F.softmax(
                logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            idx = torch.cat(
                (idx, next_token),
                dim=1
            )

        self.train()

        return idx


# --------------------------------------------------
# Loss estimation
# --------------------------------------------------

@torch.no_grad()
def estimate_loss(model):

    model.eval()

    losses = {}

    for split in ["train", "val"]:

        split_losses = torch.zeros(
            eval_iters
        )

        for k in range(eval_iters):

            X, Y = get_batch(split)

            _, loss, _ = model(X, Y)

            split_losses[k] = loss.item()

        losses[split] = split_losses.mean().item()

    model.train()

    return losses


# --------------------------------------------------
# Create model
# --------------------------------------------------

model = LSTMModel(
    vocab_size=vocab_size,
    n_embd=n_embd,
    n_layer=n_layer,
    dropout_rate=dropout_rate
).to(device)


print(
    f"Model parameters: "
    f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
)

print(f"Device: {device}")


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = AdamW(
    model.parameters(),
    lr=learning_rate
)


# --------------------------------------------------
# Learning-rate scheduler
# --------------------------------------------------

scheduler = StepLR(
    optimizer,
    step_size=100,
    gamma=0.1
)


# --------------------------------------------------
# Training
# --------------------------------------------------

best_val_loss = float("inf")


print("\nStarting training...")


for iteration in range(max_iters):

    # ----------------------------------------------
    # Evaluation
    # ----------------------------------------------

    if (
        iteration % eval_interval == 0
        or iteration == max_iters - 1
    ):

        losses = estimate_loss(model)

        print(
            f"step {iteration}: "
            f"train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )

        # Save best model

        if losses["val"] < best_val_loss:

            best_val_loss = losses["val"]

            torch.save(
                model.state_dict(),
                "best_model.pth"
            )

            print("Best model saved.")


    # ----------------------------------------------
    # Get training batch
    # ----------------------------------------------

    xb, yb = get_batch("train")


    # ----------------------------------------------
    # Forward pass
    # ----------------------------------------------

    logits, loss, _ = model(
        xb,
        yb
    )


    # ----------------------------------------------
    # Backpropagation
    # ----------------------------------------------

    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()


    # ----------------------------------------------
    # Gradient clipping
    # ----------------------------------------------

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )


    # ----------------------------------------------
    # Update weights
    # ----------------------------------------------

    optimizer.step()


    # ----------------------------------------------
    # Update learning rate
    # ----------------------------------------------

    scheduler.step()


print("\nTraining completed.")


# --------------------------------------------------
# Load best model
# --------------------------------------------------

model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location=device
    )
)

model.to(device)

print("Best model loaded.")


# --------------------------------------------------
# Generate text
# --------------------------------------------------

print("\nGenerated text:\n")


numbers = list(range(1, 6))
random.shuffle(numbers)


for i in range(5):

    context = torch.zeros(
        (1, 1),
        dtype=torch.long,
        device=device
    )

    max_tokens = 10 + numbers[i] * 5

    generated = model.generate(
        context,
        max_new_tokens=max_tokens
    )

    generated_tokens = generated[0].tolist()

    print(
        f"{i + 1}. "
        f"{decode(generated_tokens)}"
    )