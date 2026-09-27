# storyteller :: maxing out my 5060 TI to tell really good (bad) bedtime stories

this is a bit of a passion project after i got a new pc to try to really learn how LLMs work. of course, none of my networks are "large", in fact, they'd be considered very small networks. either way, the mechanisms of the networks are similar to the mechanisms of the chatbots that are prevalent in modern day times, but scaled to many orders of magnitude of data.

the project follows one question from start to finish: *how do you get a small neural network, trained from scratch on a single consumer GPU (an RTX 5060 Ti), to tell a coherent bedtime story?* it starts from tokens and a hand-built LSTM cell, moves to an LSTM language model, then to transformers, and finally to post-training with direct preference optimisation (DPO) using a local LLM as a judge.

## project goals

- [X] train lstm model, with experimentation on different layer and neuron counts
- [X] train transformer model, show dominance in performance vs lstm
- [ ] post training
    - [X] llm-as-a-judge post-training with preferred outputs
    - [ ] llm-as-a-judge post-training with metrics scoring (creativity, correctness, etc.)
- [ ] model optimisations, and transformer variants

## models

| model | parameters | context | validation loss | perplexity | weights |
| --- | --- | --- | --- | --- | --- |
| lstm (2 layers, 768 neurons each, 256 embedding) | 18.1m | 128 | 2.92 | 18.51 | `weights/20260921_lstm.pt` * |
| transformer (2 blocks, 768 feed forward, 8 heads) | 22.9m | 512 | 1.59 | 4.90 | `weights/20260925_768_768_8_8_0_0_transformer.pt` |
| transformer (4 blocks, 3024 feed forward, 8 heads) | 43.8m | 512 | 1.36 | 3.88 | `weights/20260926_3024_4x_8_4x_0_4x_transformer.pt` |
| transformer (4 blocks) + DPO | 43.8m | 512 | not measured | not measured | `weights/20260927_3024_4x_8_4x_0_4x_transformer_dpo.pt` |

the DPO model is judged on preference rather than loss: it is preferred over its base model **56.2%** of the time by the LLM judge (281/500, p = 0.0063).

\* the lstm numbers are from the one-epoch notebook run. the saved weights come from a longer 3-epoch run through `src/training/trainer.py`, which scored a slightly *worse* 2.94 validation loss and 18.88 perplexity (see [section 3](#3-an-lstm-language-model)).


## repository layout

```
src/
  configs/          dataclass configs (LSTMConfig, TrainConfig) and a json writer
  data/dataset.py   LMDataset: memory-mapped uint16 token file -> (x, y) shifted by one token
  models/
    lstm_lm.py        LSTMLangModel: embedding -> stacked nn.LSTM -> linear to vocab
    transformer_lm.py DecoderBlock + GPTNeo: pre-norm decoder-only transformer
    generate.py       autoregressive sampling (temperature / greedy) and pretty printing
  training/trainer.py train_lang_model: mixed precision loop with gradient clipping + loss log
  notebooks/
    token-embedding/01-understanding-tokens.ipynb
    lstm/01-understanding-lstms.ipynb
    lstm/02-lstm-language-model.ipynb
    transformers/01-understanding-transformers.ipynb
    transformers/02-multi-head-attention-and-deeper-layers.ipynb
    transformers/03-post-training.ipynb
    transformers/04-dpo-eval.ipynb
weights/            trained model weights (git LFS) and the retrained 10k tokeniser
```

the notebooks are the research log and should be read in the order above. the files in `src/` are the reusable pieces that were pulled out of the notebooks once they worked.



# research

## tinystories

every model here is trained on **TinyStories** (Eldan and Li, 2023), a dataset of short, synthetic children's stories written with a small vocabulary. the idea behind the dataset is that it keeps the *language* simple enough that a very small model can still learn grammar, consistency and some basic reasoning, which makes it ideal for a single GPU.

| split | stories | tokens (10k BPE) |
| --- | --- | --- |
| train | 2,119,719 | ~465m |
| validation | 21,990 | ~4.67m |

the stories are tokenised once, an `<|endoftext|>` token is appended to each story, and everything is dumped into one flat `uint16` binary file. `LMDataset` then memory-maps that file and cuts it into fixed-length windows, where the target is just the input shifted by one token:

```python
chunk = self.tokens[start:start+self.seq_len+1]
x, y = chunk[:-1], chunk[1:]
```

this means stories are packed back to back, and a window can run across the end of one story into the next (the end of text token marks the boundary). the validation split is used as the test set throughout.

## 1. tokens and embeddings

`notebooks/token-embedding/01-understanding-tokens.ipynb`

models don't eat raw text. text is first split into *tokens* (unique ids for words or parts of words), and each token id indexes a row of an *embedding matrix* $\mathbf{M} \in \mathbb{R}^{T \times M}$ for vocabulary size $T$ and embedding size $M$. that embedding is learnt during backpropagation along with the rest of the network.

using the GPT-2 tokeniser, `"The brown fox jumped over the lazy dog"` becomes `[464, 7586, 21831, 11687, 625, 262, 16931, 3290]`, where the `Ġ` prefix marks a leading space. on the way out, the network projects its hidden state back to vocabulary size and a softmax picks the next token. these two ends (embedding in, linear projection out) are shared by every model in this repo.

## 2. understanding lstms

`notebooks/lstm/01-understanding-lstms.ipynb`

before building a language model, i built an LSTM cell by hand to understand what it does. an RNN shares weights across time steps, and an LSTM adds a *cell state* $c$ which is carried through the sequence and is controlled by gates:

$$
\begin{aligned}
f, i, o &= \sigma(\cdot), \quad g = \tanh(\cdot) \quad \text{(from } W_x x_t + W_h h_{t-1}\text{)} \\
c_t &= f \odot c_{t-1} + i \odot g \\
h_t &= o \odot \tanh(c_t)
\end{aligned}
$$

the forget gate $f$ decides how much old cell state to keep, the input gate $i$ scales the update $g$, and the output gate $o$ decides how much of the cell state is exposed as $h_t$.

**experiment 1 :: remember the first value.** given a random binary sequence of length 8, predict its first element. the homebrew LSTM (hidden size 8) learns this perfectly: test loss **0.0058** and **100%** accuracy.

**experiment 2 :: lstm vs rnn on "remember the prefix".** using the built-in `nn.LSTM` and `nn.RNN` (hidden size 32), both models had to recall the first 4 bits of sequences of length 32 up to 64, trained for 2,000 epochs each. the LSTM beat the plain RNN at most sequence lengths (barring noise). the cell state gives the LSTM a path to carry the first few values all the way to the end of the sequence, whereas in the RNN that early information gets diluted with every step.

## 3. an lstm language model

`notebooks/lstm/02-lstm-language-model.ipynb`, `src/models/lstm_lm.py`, `src/training/trainer.py`

the language model is `embedding (256) -> LSTM layers -> linear (vocab)`, trained on next-token prediction with cross-entropy. the tokeniser is a 10k BPE tokeniser trained on TinyStories (`vuiseng9/bpe-10.0k-tinystories`), with 128-token windows, Adam at `1e-3` and fp16 autocast.

to measure the models, i use **perplexity**, the exponential of the average validation loss:

$$
PPL = e^{\tilde{L}} = \left(\prod_{t=1}^N P(x_t | x_{<t}) \right)^{-1/N}
$$

it can be roughly read as the number of tokens the model is "choosing between" at each step. a perfect model has perplexity 1.

| run | layers | batch | epochs | validation loss | perplexity |
| --- | --- | --- | --- | --- | --- |
| v1 | [768] | 48 | 1 | 3.3403 | 28.23 |
| v2 (wider) | [1536] | 48 | 1 | 3.3526 | 28.58 |
| v3 (deeper) | [768, 768] | 32 | 1 | **2.9185** | **18.51** |
| saved run (`trainer.py`) | [768, 768] | 32 | 3 | 2.9381 | 18.88 |

what i found:

- **wider didn't help.** doubling the single layer to 1,536 neurons gave no improvement; the training loss for v2 actually crept back up over the second half of the epoch. the original width already had enough expression.
- **deeper did help.** adding a second 768 layer took validation loss under 3 and perplexity from ~28 to ~18.5. it starts slower, but keeps learning for longer.
- **gradient norms.** for v3 i tracked the raw gradient norm and clipped it to 1. the norm slowly *increases* over training, which isn't worrying at those magnitudes and is likely a side effect of a fixed learning rate and a network that is getting more sensitive as it becomes more fine tuned.
- **more epochs didn't help much.** the same two-layer model trained for 3 epochs through `train_lang_model` (these are the saved weights) finished at 2.94, slightly worse than the single epoch notebook run. with no learning rate decay, the extra epochs didn't buy anything.

the stories have the *shape* of a story, but lose track of context almost immediately:

> Once upon a time, there was a little girl who loved to play outside. One day, she saw that his blocks there was a thief. He was scared to run away and even united. He picked up his package and ran to his mom. The end the horse wouldn't say possible skills, and the billboard treasured all right. [...]

the main failure is that the model can't hold on to *who* the story is about. a character introduced as a girl becomes "he" a sentence later, and new characters (a horse, a frog, a turkey) keep appearing. this might be because a name isn't seen often enough to learn its gender, or because the distance between the noun and the pronoun is too far for the information to survive the cell state. either way, this is what motivates moving to transformers.

## 4. a homebrew transformer

`notebooks/transformers/01-understanding-transformers.ipynb`

transformers are not recurrent: every token in the context window is processed in parallel during training, and each token can directly *attend* to every earlier token. i implemented single-head self attention by hand:

$$
\text{SelfAttention}(X) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V, \quad Q = XW_Q^T,\ K = XW_K^T,\ V = XW_V^T
$$

$Q$ is what a token is looking for, $K$ is what a token is, and $V$ is the information that gets passed on. the causal mask $M$ sets $M_{ij} = -\infty$ for $j > i$, so a token can't see the future. the block is pre-norm with residual connections:

$$
\begin{aligned}
X_{interim} &= X + \text{SelfAttention}(\text{LayerNorm}_1(X)) \\
X_{output} &= X_{interim} + \text{FeedForward}(\text{LayerNorm}_2(X_{interim}))
\end{aligned}
$$

since attention has no sense of order by itself (`gloria ate the biscuit` vs `biscuit ate the gloria`), a learnt **positional embedding** $\mathbf{p}_p$ is added to each token embedding: $\mathbf{x}_p = \mathbf{e}_p + \mathbf{p}_p$.

the first attempt was 16 blocks at width 256 (feed forward 768), roughly matching the LSTM's parameter count, with a 128-token context, trained for 3 epochs. **it was disappointing**: the recent training loss only reached **~4.52** (a perplexity of ~92), far worse than the LSTM. possible explanations are the single attention head limiting what each layer can learn, 16 narrow layers being a poor use of the parameter budget, and an untuned fixed learning rate. the validation set was never evaluated for this run, so this is training loss only.

## 5. multi-head attention and the gpt-neo setup

`notebooks/transformers/02-multi-head-attention-and-deeper-layers.ipynb`, `src/models/transformer_lm.py`

this time i followed *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* (Eldan and Li, 2023), which reports that very small GPT-Neo style models get below 1.5 loss on this dataset. the changes:

1. **a new tokeniser.** the GPT-Neo tokeniser was retrained on the TinyStories training text with a 10k vocabulary (saved to `weights/gpt-neo-10k-tokeniser`), and the dataset re-tokenised.
2. **pytorch's `nn.MultiheadAttention`** instead of the homebrew head. each of $h$ heads gets its own projections into $d_h = d/h$ dimensions, the outputs are concatenated back to $d$, then mixed with a linear layer:
   $$H_r = \text{softmax}\left(\frac{Q_rK_r^T}{\sqrt{d_h}} + C\right)V_r$$
3. **a 512-token context window**, 768-dimensional embeddings, GELU feed forward and a final layer norm, like the paper's GPT-Neo 33M model (which uses 4 blocks, 16 heads and a 3072 feed forward).
4. **AdamW** (`lr = 1e-3`, weight decay `0.01`), gradient clipping at 1 and batch size 32.

### 5a. two blocks

to fit comfortably in memory, the first model uses **2 blocks, 8 heads and a 768 feed forward**, trained for 2 epochs with fp16 and a gradient scaler.

- training loss: 1.58
- **validation loss: 1.5898, perplexity: 4.90**

going from a perplexity of ~19 to ~5 is a massive jump. rather than "choosing" from roughly 19 tokens, the model is on average choosing from 5. more importantly, it holds on to its characters:

> Once upon a time, there was a little girl named Lily. Lily was very happy and loved Christmas. She dreamed of the big tree she had seen and the bright lights in the sky. One day, she went with her mommy to get dressed. Her mommy took her to the store and they bought yummy bananas. [...] And from that day on Lily always asked for the yummy bananas.

Lily is referred to as she/her throughout, and the bananas introduced in the middle of the story are still there at the end.

### 5b. four blocks, wider feed forward

next, deeper *and* wider: **4 blocks, 8 heads and a 3024 feed forward**, trained for 3 epochs with `bfloat16` autocast. bfloat16 has the same range as fp32 (with less precision), so the gradient scaler is no longer needed.

- training loss: 1.34
- **validation loss: 1.3560, perplexity: 3.88**

this is a further drop of ~0.24 in loss and ~1.0 in perplexity, and is very close to what the paper reports. the paper likely spent more effort tuning training parameters, which is possible on my hardware but tedious. at a temperature of 0.75:

> A long, long time ago, there was a big, hairy bear. He was very angry because someone was trying to cut down his tree. He had a lot of hope to get his tree back. But then, he saw a little bird fly down and land on his shoulder. The bird had a hurt wing and couldn't fly. The bear felt sorry for the bird and decided to help. He gently picked up the bird and put it on a branch of the tree. The bird felt better and flew away. The bear was happy to have helped the little bird.

### where the parameters live

| model | embeddings (token + position) | blocks | output layer | total |
| --- | --- | --- | --- | --- |
| lstm [768, 768] | 2.6m | 7.9m (lstm layers) | 7.7m | 18.1m |
| transformer, 2 blocks | 8.1m | 7.1m | 7.7m | 22.9m |
| transformer, 4 blocks | 8.1m | 28.1m | 7.7m | 43.8m |

with a 10k vocabulary, a large share of every model is the embedding and output layers. the jump from 2 to 4 blocks is almost entirely feed forward weights: each 3024-wide feed forward holds ~4.6m parameters, compared to ~1.2m at 768.

## 6. post-training with dpo and an llm judge

`notebooks/transformers/03-post-training.ipynb`

a 1.36 loss is good, but the stories were still slightly off and largely uncreative. the next step was **post-training**: generate several stories for the same prompt, let a bigger local LLM pick the best one, and nudge the model towards the stories it prefers.

### generating the preference data

1. sample **2,500 stories** from the validation set and take each one's first sentence (up to the first `.`, `?` or `!`) as the prompt.
2. generate **4 completions** per prompt from the 4-block model at temperature 1 (up to 300 new tokens), so there is real variation between them.
3. ask **`qwen3.5:4b`** (run locally through `ollama`, temperature 0, no thinking, structured json output) to pick the best completion. the judge prompt asks it to rank, in order of importance:
    1. *coherence*: same characters, no contradictions or repetition, stays on topic
    2. *grammar*: correct sentences and a proper ending
    3. *creativity*: something interesting or warm happens that a young child would enjoy

   it also states that simple words are fine for ages 3-4, that a plain coherent story beats a creative one that falls apart, and that length shouldn't matter.

early runs showed that the judge favoured the first completion it saw, so the completions are shuffled before being shown. even after shuffling, the choices weren't quite uniform:

| completion id | 0 | 1 | 2 | 3 | invalid (4) |
| --- | --- | --- | --- | --- | --- |
| times chosen | 736 | 521 | 647 | 590 | 6 |

the 6 invalid answers are dropped. every other prompt gives 3 (chosen, rejected) pairs, for **7,482 training pairs**.

### the dpo objective

the log probability of a completion $y$ given a prompt $x$ is the sum of its token log probabilities:

$$
\log \pi_\theta(y | x) = \sum_{t=1}^m \log \pi_\theta(y_t | x, y_{<t})
$$

only the completion tokens are counted, not the prompt. with the trained model $\pi_e$ and a frozen copy of the original $\pi_x$ as a reference, the margin is

$$
u = \beta\left([\log\pi_e(y_{accepted}) - \log\pi_e(y_{rejected})] - [\log\pi_x(y_{accepted}) - \log\pi_x(y_{rejected})]\right)
$$

and the loss is $L = -\log\sigma(u)$. minimising it rewards the model for becoming *relatively* more likely than the reference to write the accepted story over the rejected one. the reference term stops the model from drifting too far from what it learnt in pre-training.

### training

| setting | value |
| --- | --- |
| $\beta$ | 0.1 |
| learning rate | `5e-6` (AdamW), much lower than pre-training since the model is already trained |
| batch | 16 pairs, via gradient accumulation |
| gradient clipping | 1.0 |
| planned epochs | 4 |

the "accuracy" below is the share of pairs where $u > 0$, i.e. the model prefers the chosen story more strongly than the reference does. the loss starts at $\ln 2 \approx 0.693$, where the two models are identical.

| point in training | loss (last 200 pairs) | accuracy |
| --- | --- | --- |
| epoch 0, start | 0.6915 | 0.57 |
| epoch 0, end | 0.6029 | 0.755 |
| epoch 1, start | 0.5056 | 0.905 |
| epoch 1, end | 0.4684 | 0.865 |

the training log stops after the second epoch, where the notebook kernel crashed, so the log doesn't show whether epochs 3 and 4 ran before the weights in `weights/20260927_3024_4x_8_4x_0_4x_transformer_dpo.pt` were saved. from epoch 1 onwards the model is also seeing pairs it has already trained on, so these accuracies measure fit rather than generalisation. that is what the next notebook is for.

## 7. does dpo generalise?

`notebooks/transformers/04-dpo-eval.ipynb`

to check, i sampled **500 new prompts** from the validation set and generated one completion from the original model and one from the DPO model for each. the same judge (with the same prompt and shuffling) picked the better one.

| | result |
| --- | --- |
| DPO model preferred | **281 / 500 (56.2%)** |
| two-sided binomial test vs 50% | p = 0.0063 |
| 95% confidence interval | 51.73% to 60.6% |

the DPO model is preferred significantly more often at $\alpha = 5\%$. this doesn't prove that its stories are more creative or more grammatical; it could just be absorbing Qwen 3.5 4b's biases. that said, Qwen's depth and data dominate this model, so distilling some of its judgement (and bias) seems like a good trade.

qualitatively, the post-trained model stays on track more. prompted with `Connor was an angry boy.`:

> **original ::** Connor was an angry boy. He had seen it and he was embarrassed. "You should,' he said. you should always be a good young boy, no matter what," stayed his father from then on. From then on, Frian was a very thoughtful mother!

> **DPO ::** Connor was an angry boy. "Roar!" he exclaimed. "What are you honking for? Juice Connor!" Claire was surprised. She had never met a car before. [...] Soon they were all laughing and playing together. It was a great day because the boy was always clapping and Ellie was glowing with joy.

the original model makes up a new story (and a new character, "Frian") out of thin air, while the DPO model at least continues from the angry boy. both models still struggle with anything more novel. asked about a boy who loves baseball, the DPO model mixes knowledge from different sports ("he had never driven a ball", "he started to run and kick the ball").

my conclusion is that DPO does help produce more coherent stories, and done at a larger scale, post-training would likely help align the model towards more interesting and vibrant stories.

---

## results summary

| stage | model | validation loss | perplexity | takeaway |
| --- | --- | --- | --- | --- |
| 3 | lstm, 1 x 768 | 3.34 | 28.2 | tells something story-shaped |
| 3 | lstm, 1 x 1536 | 3.35 | 28.6 | width doesn't help |
| 3 | lstm, 2 x 768 | 2.92 | 18.5 | depth helps, but characters still drift |
| 4 | homebrew transformer, 16 x 256, 1 head | ~4.52 (train) | ~92 (train) | a naive transformer loses to the lstm |
| 5a | gpt-neo style, 2 blocks | 1.59 | 4.90 | multi-head attention + 512 context changes everything |
| 5b | gpt-neo style, 4 blocks, 3024 ff | 1.36 | 3.88 | close to the TinyStories paper |
| 6-7 | 5b + DPO | n/a | n/a | preferred by the judge 56.2% of the time (p = 0.0063) |

## caveats

- **the lstm and transformer losses use different tokenisers** (`vuiseng9/bpe-10.0k-tinystories` vs the retrained GPT-Neo tokeniser) and different context lengths (128 vs 512). both vocabularies are 10k and both produce ~465m training tokens, so the numbers are roughly comparable, but it isn't a perfectly controlled comparison.
- **the validation split doubles as the test set**, and it is also where the DPO training and evaluation prompts were sampled from. the 500 evaluation prompts can overlap with the 2,500 DPO training prompts (many prompts are just "Once upon a time, there was a little girl named Lily."), although the completions themselves are freshly sampled.
- **the judge is a single small model.** in the evaluation, the original model's completion is always labelled `Completion 0` and the DPO model's `Completion 1`. the order is shuffled, but the labels aren't. during data generation the judge picked id 0 slightly more often than chance, so if anything a label bias would work *against* the DPO model.
- **each evaluation prompt gets one sample per model** at temperature 1, so part of the 56.2% is sampling noise; this is what the confidence interval accounts for.

## running it yourself

the notebooks aren't a packaged pipeline; some paths are hardcoded (sory).

- **data**: the TinyStories parquet files (train and validation splits) from hugging face.
- **python packages**: `torch`, `transformers`, `numpy`, `pandas` (with a parquet engine), `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, plus `ollama` and `pydantic` for the judge.
- **judge**: a running [ollama](https://ollama.com) server with `qwen3.5:4b` pulled.
- **weights**: tracked with git LFS, so run `git lfs pull` after cloning.

loading the best model and telling a story:

```python
import torch
from transformers import AutoTokenizer
from src.models.transformer_lm import GPTNeo
from src.models.generate import generate, pretty_print

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokeniser = AutoTokenizer.from_pretrained('weights/gpt-neo-10k-tokeniser')

model = GPTNeo(4, 10_000, 768, 512, [3024] * 4, [8] * 4, dropout=[0] * 4).to(device)
model.load_state_dict(torch.load('weights/20260927_3024_4x_8_4x_0_4x_transformer_dpo.pt', map_location=device))

pretty_print(generate('Once upon a time,', model, tokeniser, device, temperature=0.75))
```

## references

- Eldan, R. and Li, Y. (2023). *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?*
- Rafailov, R. et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.*
- Vaswani, A. et al. (2017). *Attention Is All You Need.*
- Hochreiter, S. and Schmidhuber, J. (1997). *Long Short-Term Memory.*
