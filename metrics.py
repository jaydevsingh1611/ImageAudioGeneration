import torch
import torch.nn.functional as F

def perplexity(model, tokenizer, true_captions, img_paths):
    total_loss = 0.0
    total_tokens = 0

    for true_caption, img_path in zip(true_captions, img_paths):
        predicted_caption = model_predict(img_path, model, path=True)
        true_tokens = tokenizer.tokenize(true_caption)
        predicted_tokens = tokenizer.tokenize(predicted_caption)
        true_indices = tokenizer.convert_tokens_to_ids(true_tokens)
        predicted_indices = tokenizer.convert_tokens_to_ids(predicted_tokens)
        true_tensor = torch.tensor(true_indices).unsqueeze(0)
        predicted_tensor = torch.tensor(predicted_indices).unsqueeze(0)
        logits = model(true_tensor)
        loss = F.cross_entropy(logits, predicted_tensor)
        total_loss += loss.item()
        total_tokens += len(true_tokens)

    return 2 ** (total_loss / total_tokens)