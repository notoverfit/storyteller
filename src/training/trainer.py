import torch.nn as nn
import pandas as pd
import torch

from typing import Literal
from torch.utils.data import DataLoader
from torch.optim import Optimizer

def train_lang_model(
    model: nn.Module, 
    loss_fn: nn.Module, 
    optim: Optimizer, 
    epochs: int, 
    tr_loader: DataLoader,
    device: torch.device,
    max_norm=float('inf'),
    progress=True
) -> pd.DataFrame:
    if epochs < 1:
        raise Exception('train_model :: epochs should be greater than or equal to 1.')

    # report every 1% of progress, or every batch if < 100 batches
    batch_flag = max(1, len(tr_loader) // 100)
    total_loss = 0
    tokens = 0
    res = []

    if progress: print('train_lang_model :: starting training...')
    for epoch in range(epochs):
        for batch_idx, (X_batch, y_batch) in enumerate(tr_loader, start=1):
            X_batch = X_batch.to(device, dtype=torch.long, non_blocking=True)
            y_batch = y_batch.to(device, dtype=torch.long, non_blocking=True)

            optim.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(X_batch)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), y_batch.reshape(-1))

            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optim.step()

            # metrics computation
            n_tokens = y_batch.numel()
            total_loss += loss.detach().item() * n_tokens
            tokens += n_tokens
            avg_loss = total_loss / tokens

            res.append({ 'i': batch_idx, 'avg_loss': avg_loss, 'grad_norm': grad_norm })

            # print update if required
            if batch_idx % batch_flag == 0 and progress:
                progress_pct = 100 * batch_idx / len(tr_loader)
                print(f'train_lang_model :: epoch {epoch} : batch {progress_pct:.1f}% : avg_loss {avg_loss:.4f}')

    return pd.DataFrame(res)