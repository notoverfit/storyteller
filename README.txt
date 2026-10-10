really bad bedtime stories made by a micro transformer
======================================================

small language models trained from scratch on TinyStories with one consumer
GPU (an RTX 5060 Ti): an LSTM, two transformers, and the best transformer
post-trained with direct preference optimisation (DPO), using a local llm as
the judge.

the best model has 43.8m parameters and a validation loss of 1.36 (3.88
perplexity). after DPO, the judge prefers its stories over the original
model's 56.2% of the time. a sample at a temperature of 0.75:

    A long, long time ago, there was a big, hairy bear. He was very angry
    because someone was trying to cut down his tree. He had a lot of hope to
    get his tree back. But then, he saw a little bird fly down and land on
    his shoulder. The bird had a hurt wing and couldn't fly. The bear felt
    sorry for the bird and decided to help. He gently picked up the bird and
    put it on a branch of the tree. The bird felt better and flew away. The
    bear was happy to have helped the little bird.


models
------

model                            parameters  context  val loss  perplexity
lstm, 2 layers of 768            18.1m       128      2.92      18.51
transformer, 2 blocks, 768 ff    22.9m       512      1.59      4.90
transformer, 4 blocks, 3024 ff   43.8m       512      1.36      3.88
the 4 block transformer + DPO    43.8m       512      not measured

the weights are in weights/ and tracked with git LFS, so run `git lfs pull`
after cloning. the retrained 10k tokeniser the transformers use is in
weights/gpt-neo-10k-tokeniser.

the lstm numbers are from a one epoch run, whereas the saved weights in weights/
are from a 3 epoch run which produced slightly worse results.

layout
------

src/models/     the lstm, the transformer and the sampling code
src/training/   the training loop
src/data/       the memory-mapped token dataset
src/configs/    dataclass configs
src/notebooks/  where the research actually happened; read them in order
                (token-embedding, lstm, transformers)
weights/        trained weights and the tokeniser


running it yourself
-------------------

the notebooks are not a packaged pipeline. you will need the TinyStories
parquet files from hugging face, the packages in requirements.txt, and for
the judge, a running ollama server with qwen3.5:4b pulled. copy .env.example
to .env and set TINYSTORIES_DIR to the folder holding your data.

to load the best model and get a story out of it:

    import torch
    from transformers import AutoTokenizer
    from src.models.transformer_lm import GPTNeo
    from src.models.generate import generate, pretty_print

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokeniser = AutoTokenizer.from_pretrained('weights/gpt-neo-10k-tokeniser')

    model = GPTNeo(4, 10_000, 768, 512, [3024] * 4, [8] * 4, dropout=[0] * 4).to(device)
    model.load_state_dict(torch.load('weights/20260927_3024_4x_8_4x_0_4x_transformer_dpo.pt', map_location=device))

    pretty_print(generate('Once upon a time,', model, tokeniser, device, temperature=0.75))


references
----------

- Eldan and Li (2023). TinyStories: How Small Can Language Models Be and
  Still Speak Coherent English?
- Rafailov et al. (2023). Direct Preference Optimization: Your Language Model
  is Secretly a Reward Model.
- Vaswani et al. (2017). Attention Is All You Need.
- Hochreiter and Schmidhuber (1997). Long Short-Term Memory.
