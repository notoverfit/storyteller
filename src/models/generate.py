import torch
import torch.nn as nn
import textwrap
from transformers import AutoTokenizer

@torch.no_grad()
def generate(prompt: str, model: nn.Module, tokeniser: AutoTokenizer, device, temperature: float = 1.0, max_new_tokens: int = 300, context_length: int = 512) -> str:
    assert temperature >= 0

    model.eval()
    prompt_tokens = tokeniser.encode(prompt)
    prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    prompt_length = prompt_tokens.size(0)

    while True:
        # adds a dimension of size 1 to the context, dummying the batch variable
        # then squeeze 0 removes that batch dimension
        context = prompt_tokens[-context_length:].unsqueeze(0)
        model_logits = model(context).squeeze(0)[-1]

        # special case for greedy
        if temperature > 0:
            probs = torch.softmax(model_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = model_logits.argmax().unsqueeze(0)

        # check if need to break
        if next_token.item() == tokeniser.eos_token_id: break

        # append the token id directly rather than decoding and re-encoding the whole string
        prompt_tokens = torch.cat([prompt_tokens, next_token])

        if prompt_tokens.size(0) - prompt_length >= max_new_tokens: break

    # decode once at the end
    return prompt + tokeniser.decode(prompt_tokens[prompt_length:].tolist())

def clean(text: str):
    return " ".join(text.split())

def pretty_print(text: str, width: int = 80):
    print(textwrap.fill(clean(text), width=width))