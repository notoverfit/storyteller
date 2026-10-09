# really bad bedtime stories made by a micro transformer

how small can a model be, trained from scratch on one consumer GPU (my RTX 5060 Ti), and still tell a bedtime story that makes sense?

a 43.8m parameter transformer gets to **1.36 validation loss (3.88 perplexity)** on TinyStories, and after DPO it is preferred over its former self **56.2%** of the time. here is one of its stories, at a temperature of 0.75:

> A long, long time ago, there was a big, hairy bear. He was very angry because someone was trying to cut down his tree. He had a lot of hope to get his tree back. But then, he saw a little bird fly down and land on his shoulder. The bird had a hurt wing and couldn't fly. The bear felt sorry for the bird and decided to help. He gently picked up the bird and put it on a branch of the tree. The bird felt better and flew away. The bear was happy to have helped the little bird.

## project goals

- [X] train lstm model, with experimentation on different layer and neuron counts
- [X] train transformer model, show dominance in performance vs lstm
- [ ] post training
    - [X] llm-as-a-judge post-training with preferred outputs
    - [ ] llm-as-a-judge post-training with metrics scoring (creativity, correctness, etc.)
- [X] model optimisations, and transformer variants

## models

all weights live in `weights/` and are tracked with git LFS, so run `git lfs pull` after cloning. the file names follow `date_feedforward_heads_dropout_model.pt`.

| model | parameters | context | validation loss | perplexity | weights |
| --- | --- | --- | --- | --- | --- |
| lstm (2 layers, 768 neurons each, 256 embedding) | 18.1m | 128 | 2.92 | 18.51 | `weights/20260921_lstm.pt` * |
| transformer (2 blocks, 768 feed forward, 8 heads) | 22.9m | 512 | 1.59 | 4.90 | `weights/20260925_768_768_8_8_0_0_transformer.pt` |
| transformer (4 blocks, 3024 feed forward, 8 heads) | 43.8m | 512 | 1.36 | 3.88 | `weights/20260926_3024_4x_8_4x_0_4x_transformer.pt` |
| transformer (4 blocks) + DPO | 43.8m | 512 | not measured | not measured | `weights/20260927_3024_4x_8_4x_0_4x_transformer_dpo.pt` |

the retrained 10k tokeniser that the transformers use is in `weights/gpt-neo-10k-tokeniser`.

the DPO model is judged on preference rather than loss; the llm judge prefers it over its base model 281 times out of 500 (p = 0.0063).

\* the lstm numbers are from the one epoch notebook run. the saved weights are from a longer 3 epoch run through `src/training/trainer.py`, which actually came out slightly *worse* at 2.94 loss and 18.88 perplexity (more on this in [section 3](#3-an-lstm-language-model)).

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

the notebooks are where the research actually happened, and should be read in the order above. the files in `src/` are the bits i pulled out of the notebooks once they worked.

# research

## tinystories

everything here is trained on **TinyStories** (Eldan and Li, 2023), a dataset of short, synthetic children's stories written with a small vocabulary. the language is kept simple enough that a very small model can still learn grammar, consistency and a bit of reasoning, which is exactly what i need with one GPU.

| split | stories | tokens (10k BPE) |
| --- | --- | --- |
| train | 2,119,719 | ~465m |
| validation | 21,990 | ~4.67m |

the stories are tokenised once, an `<|endoftext|>` token is added to the end of each story, and the whole lot is dumped into one flat `uint16` binary file. `LMDataset` memory-maps that file and cuts it into fixed length windows, where the target is just the input shifted along by one token:

```python
chunk = self.tokens[start:start+self.seq_len+1]
x, y = chunk[:-1], chunk[1:]
```

so stories are packed back to back, and a window can run off the end of one story and into the next (the end of text token marks the boundary). i use the validation split as the test set throughout.

## 1. tokens and embeddings

`notebooks/token-embedding/01-understanding-tokens.ipynb`

machine learning models generally do not ingest raw text. text is turned into *tokens*, which are essentially unique ids for words (or parts of words), and each token id indexes a row of an embedding matrix $\mathbf{M} \in \mathbb{R}^{T \times M}$ for vocabulary size $T$ and embedding size $M$. the embedding is learnt during backpropagation with the rest of the network.

with the GPT-2 tokeniser, `"The brown fox jumped over the lazy dog"` becomes `[464, 7586, 21831, 11687, 625, 262, 16931, 3290]`. on the way out, the network projects back up to vocabulary size and a softmax picks the next token. every model in this repo has these same two ends; an embedding in, and a linear projection out.

## 2. understanding lstms

`notebooks/lstm/01-understanding-lstms.ipynb`

`pytorch` has an `LSTM` pre-baked, but before using it i wanted to build my own cell to understand what it does. an RNN shares weights across each recurrence, and an LSTM adds a *cell state* $c$ which is carried through the sequence and controlled by gates:

$$
\begin{aligned}
f, i, o &= \sigma(\cdot), \quad g = \tanh(\cdot) \quad \text{(from } W_x x_t + W_h h_{t-1}\text{)} \\
c_t &= f \odot c_{t-1} + i \odot g \\
h_t &= o \odot \tanh(c_t)
\end{aligned}
$$

the forget gate $f$ decides how much of the old cell state to keep, the input gate $i$ scales the update $g$, and the output gate $o$ decides how much of the cell state is shown as $h_t$.

**experiment 1 :: remember the first value.** given a random binary sequence of length 8, predict its first element. the homebrew LSTM (hidden size 8) learns the rule perfectly, with a test loss of **0.0058** and **100%** accuracy.

**experiment 2 :: lstm vs rnn on "remember the prefix".** using the pre-baked `nn.LSTM` and `nn.RNN` (hidden size 32), both models had to recall the first 4 bits of sequences of length 32 up to 64, with 2,000 epochs each. the LSTM dominates the RNN at most sequence lengths (barring noise). the cell state gives it a way to carry those first few values to the end of the sequence, whereas the RNN has no mechanism for this and the early information gets diluted.

## 3. an lstm language model

`notebooks/lstm/02-lstm-language-model.ipynb`, `src/models/lstm_lm.py`, `src/training/trainer.py`

the language model is `embedding (256) -> LSTM layers -> linear (vocab)`, trained to predict the next token with cross-entropy. it uses a 10k BPE tokeniser trained on TinyStories (`vuiseng9/bpe-10.0k-tinystories`), 128 token windows, Adam at `1e-3` and fp16 autocast.

how do we measure it? i use **perplexity**, which is the exponential of the average validation loss:

$$
\mathrm{PPL}
= e^{L}
= \left(
\prod_{t=1}^{N} P(x_t \mid x_{1:t-1})
\right)^{-1/N}
$$

it can be roughly considered the number of tokens that the model is picking from at each prediction. a model that always predicts the next word correctly has a perplexity of 1.

| run | layers | batch | epochs | validation loss | perplexity |
| --- | --- | --- | --- | --- | --- |
| v1 | [768] | 48 | 1 | 3.3403 | 28.23 |
| v2 (wider) | [1536] | 48 | 1 | 3.3526 | 28.58 |
| v3 (deeper) | [768, 768] | 32 | 1 | **2.9185** | **18.51** |
| saved run (`trainer.py`) | [768, 768] | 32 | 3 | 2.9381 | 18.88 |

- **wider did nothing.** i noticed during tuning that wider networks seemed to do better, so i doubled the layer to 1,536 neurons. no improvement; the training loss even crept back up over the second half of the epoch. the original width already had enough expression.
- **deeper helped.** a second 768 layer got the loss under 3 and the perplexity from ~28 to ~18.5. it is slower to start, but learns for longer.
- **gradient norms.** for v3 i tracked the gradient norm and clipped it to 1 (a rule of thumb, not necessarily optimal). the norm slowly *increases* over training. this isn't of great concern at those magnitudes, and is likely a side effect of the fixed learning rate and the network getting more sensitive as it becomes more fine tuned.
- **more epochs didn't help either.** i expected the two layer model to keep improving with more epochs, so i trained it for 3 through `train_lang_model` (these are the saved weights). it finished at 2.94, slightly worse than the one epoch run. with no learning rate decay, the extra epochs bought nothing.

how does it fare at telling a story? it has the *shape* of one, but loses the plot almost immediately:

> Once upon a time, there was a little girl who loved to play outside. One day, she saw that his blocks there was a thief. He was scared to run away and even united. He picked up his package and ran to his mom. The end the horse wouldn't say possible skills, and the billboard treasured all right. [...]

the first sentence is okay, and then we get horses, frogs and a turkey. the model can't hold on to *who* the story is about; a girl becomes "he" one sentence later. perhaps a name doesn't come up often enough to learn its gender, or the distance between the noun and the pronoun is too far for the information to survive the cell state. either way, this is why i moved on to transformers.

## 4. a homebrew transformer

`notebooks/transformers/01-understanding-transformers.ipynb`

transformers are *not* recurrent. every token in the context window is processed in parallel during training, and each token can directly *attend* to every token before it. again there is a pre-baked version in `torch`, and again i wrote my own first (single head only):

$$
\text{SelfAttention}(X) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V, \quad Q = XW_Q^T,\ K = XW_K^T,\ V = XW_V^T
$$

$Q$ is what a token is looking for, $K$ is what a token is, and $V$ is the info that gets passed on. the causal mask $M$ sets $M_{ij} = -\infty$ for $j > i$, so a token cannot see the future. the block normalises before each step and adds its input back on (the residual connection):

$$
\begin{aligned}
X_{interim} &= X + \text{SelfAttention}(\text{LayerNorm}_1(X)) \\
X_{output} &= X_{interim} + \text{FeedForward}(\text{LayerNorm}_2(X_{interim}))
\end{aligned}
$$

attention has no idea about order by itself (`gloria ate the biscuit` and `biscuit ate the gloria` mean different things), so a learnt **positional embedding** $\mathbf{p}_p$ is added to each token embedding: $\mathbf{x}_p = \mathbf{e}_p + \mathbf{p}_p$.

i went for 16 blocks at width 256 (feed forward 768) to roughly match the LSTM's parameter count, with a 128 token context and 3 epochs of training. **it was not good.** the training loss only got to **~4.52**, or a perplexity of ~92, which is far worse than the LSTM. a single attention head may limit what each layer can learn, 16 narrow layers may be a poor use of the parameters, and the fixed learning rate was never tuned; these are possible explanations, not confirmed causes. i also never evaluated the validation set for this run, so this is training loss only.

## 5. multi-head attention and the gpt-neo setup

`notebooks/transformers/02-multi-head-attention-and-deeper-layers.ipynb`, `src/models/transformer_lm.py`

this time i took direction from *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* (Eldan and Li, 2023), which describes very small GPT-Neo style models getting below 1.5 loss on this dataset. what changed:

1. **a new tokeniser.** the GPT-Neo tokeniser was retrained on the TinyStories training text with a 10k vocabulary (saved to `weights/gpt-neo-10k-tokeniser`), and the data set re-tokenised.
2. **pytorch's `nn.MultiheadAttention`** instead of my homebrew head. each of the $h$ heads learns its own projections into $d_h = d/h$ dimensions, the heads are concatenated back to $d$, and a linear layer mixes them:
   $$H_r = \text{softmax}\left(\frac{Q_rK_r^T}{\sqrt{d_h}} + C\right)V_r$$
3. **a 512 token context window**, 768 dimension embeddings, a GELU feed forward and a final layer norm, like the paper's GPT-Neo 33M model (which has 4 blocks, 16 heads and a 3072 feed forward).
4. **AdamW** (`lr = 1e-3`, weight decay `0.01`), gradient clipping at 1 and a batch size of 32.

### 5a. two blocks

to fit within my GPU's memory, the first model has **2 blocks, 8 heads and a 768 feed forward**, trained for 2 epochs with fp16 and a gradient scaler.

- training loss: 1.58
- **validation loss: 1.5898, perplexity: 4.90**

incredible stuff; the perplexity goes from ~19 to ~5. instead of "randomly" (not exactly correct, but bear with me) choosing from 19 words, the model is on average choosing from 5. better yet, it remembers its characters:

> Once upon a time, there was a little girl named Lily. Lily was very happy and loved Christmas. She dreamed of the big tree she had seen and the bright lights in the sky. One day, she went with her mommy to get dressed. Her mommy took her to the store and they bought yummy bananas. [...] And from that day on Lily always asked for the yummy bananas.

Lily is she/her the whole way through, and the bananas from the middle of the story are still there at the end. compare that with the LSTM's turkey.

### 5b. four blocks, wider feed forward

we can go deeper (and wider!): **4 blocks, 8 heads and a 3024 feed forward**, trained for 3 epochs with `bfloat16` autocast. bfloat16 has the range of fp32 with less precision, which removes the need for the scaler.

- training loss: 1.34
- **validation loss: 1.3560, perplexity: 3.88**

that is another ~0.24 off the loss and ~1.0 off the perplexity, which gets very close to the paper. they likely spent more effort optimising training parameters; with my compute this is possible, but quite tedious, and i am fairly satisfied with these results. the bear story at the top of this README is from this model.

### where the parameters live

| model | embeddings (token + position) | blocks | output layer | total |
| --- | --- | --- | --- | --- |
| lstm [768, 768] | 2.6m | 7.9m (lstm layers) | 7.7m | 18.1m |
| transformer, 2 blocks | 8.1m | 7.1m | 7.7m | 22.9m |
| transformer, 4 blocks | 8.1m | 28.1m | 7.7m | 43.8m |

with a 10k vocabulary, a big share of every model is just the embedding and output layers. going from 2 to 4 blocks is almost all feed forward weights; each 3024 wide feed forward holds ~4.6m parameters, compared to ~1.2m at 768.

## 6. post-training with dpo and an llm judge

`notebooks/transformers/03-post-training.ipynb`

1.36 is a pretty good loss, but the stories were still slightly off and largely uncreative. so the next idea was post-training: generate a few stories from the same prompt, get a bigger local llm to choose the best one, and nudge the weights towards the stories it prefers.

### generating the preference data

1. sample **2,500 stories** from the validation set and take the first sentence of each (up to the first `.`, `?` or `!`) as the prompt.
2. generate **4 completions** per prompt from the 4 block model at temperature 1 (up to 300 new tokens), to allow for some variation between them.
3. ask **`qwen3.5:4b`** (run locally through `ollama`, temperature 0, no thinking, structured json output) to choose the best completion. the judge prompt asks for, in order of importance:
    1. *coherence*: same characters, no contradictions or repetition, stays on topic
    2. *grammar*: correct sentences and a proper ending
    3. *creativity*: something interesting or warm happens that a young child would enjoy

   it is also told that simple words are fine for ages 3-4, that a plain coherent story beats a creative one that falls apart, and that length shouldn't matter.

initial runs showed the judge was biased towards the first completion it saw, so the completions are shuffled around. even after shuffling, the choices aren't quite uniform:

| completion id | 0 | 1 | 2 | 3 | invalid (4) |
| --- | --- | --- | --- | --- | --- |
| times chosen | 736 | 521 | 647 | 590 | 6 |

the 6 invalid answers are dropped. every other prompt gives 3 (chosen, rejected) pairs, for **7,482 training pairs**.

### the dpo objective

the log probability of a completion $y$ given a prompt $x$ is the sum of the log probabilities of its tokens:

$$
\log \pi_\theta(y | x) = \sum_{t=1}^m \log \pi_\theta(y_t | x, y_{\lt t})
$$

only the completion tokens count, not the prompt. with the model being trained $\pi_e$ and a frozen copy of the original $\pi_x$ as a reference, the margin is

$$
u = \beta\left([\log\pi_e(y_{accepted}) - \log\pi_e(y_{rejected})] - [\log\pi_x(y_{accepted}) - \log\pi_x(y_{rejected})]\right)
$$

and the loss is $L = -\log\sigma(u)$. thus, in a minimisation, the model is "encouraged" to favour the accepted story over the rejected one by more than the reference does. the reference term stops it from drifting too far from what it learnt in pre-training.

### training

| setting | value |
| --- | --- |
| $\beta$ | 0.1 |
| learning rate | `5e-6` (AdamW) |
| batch | 16 pairs, via gradient accumulation |
| gradient clipping | 1.0 |
| planned epochs | 4 |

the learning rate is much lower than in pre-training, as this is *already* a trained model and a higher one can overshoot. the "accuracy" below is the share of pairs where $u > 0$. the loss starts at $\ln 2 \approx 0.693$, where the two models are identical.

| point in training | loss (last 200 pairs) | accuracy |
| --- | --- | --- |
| epoch 0, start | 0.6915 | 0.57 |
| epoch 0, end | 0.6029 | 0.755 |
| epoch 1, start | 0.5056 | 0.905 |
| epoch 1, end | 0.4684 | 0.865 |

the log stops after the second epoch because the notebook kernel crashed, so i can't show whether epochs 3 and 4 ran before the weights in `weights/20260927_3024_4x_8_4x_0_4x_transformer_dpo.pt` were saved. from epoch 1 onwards the model is also seeing pairs it has already trained on, so these accuracies say more about fit than generalisation. that is what the next notebook is for.

## 7. does dpo generalise?

`notebooks/transformers/04-dpo-eval.ipynb`

i took another sample of **500 prompts** from the validation set, generated one completion from the original model and one from the DPO model for each, and had the same judge (same prompt, same shuffling) choose between them.

| | result |
| --- | --- |
| DPO model preferred | **281 / 500 (56.2%)** |
| two-sided binomial test vs 50% | p = 0.0063 |
| 95% confidence interval | 51.73% to 60.6% |

the DPO model is preferred, and this is significant at $\alpha = 5\%$. this does not mean that its stories are more creative or more grammatically correct; we could just be teaching it Qwen 3.5 4b's biases. either way, the Qwen model's depth and data dominate my transformer, so distilling some information and some bias from it sounds like a good thing to me?

is there a *qualitative* difference? i think so. the post-trained model stays on track more. prompted with `Connor was an angry boy.`:

> **original ::** Connor was an angry boy. He had seen it and he was embarrassed. "You should,' he said. you should always be a good young boy, no matter what," stayed his father from then on. From then on, Frian was a very thoughtful mother!

> **DPO ::** Connor was an angry boy. "Roar!" he exclaimed. "What are you honking for? Juice Connor!" Claire was surprised. She had never met a car before. [...] Soon they were all laughing and playing together. It was a great day because the boy was always clapping and Ellie was glowing with joy.

the original model sort of hallucinates a story (and a "Frian") out of thin air, whilst the DPO model at least carries on with the angry boy. both still struggle with anything more novel. asked about a boy who loves baseball, the DPO model amalgamates a few different sports ("he had never driven a ball", "he started to run and kick the ball").

so i think DPO does help with coherence. done at a larger scale, post-training would likely help align the model towards more interesting and vibrant stories.

---

## results summary

| stage | model | validation loss | perplexity | takeaway |
| --- | --- | --- | --- | --- |
| 3 | lstm, 1 x 768 | 3.34 | 28.2 | tells something story-shaped |
| 3 | lstm, 1 x 1536 | 3.35 | 28.6 | width doesn't help |
| 3 | lstm, 2 x 768 | 2.92 | 18.5 | depth helps, but characters still drift |
| 4 | homebrew transformer, 16 x 256, 1 head | ~4.52 (train) | ~92 (train) | my naive transformer loses to the lstm |
| 5a | gpt-neo style, 2 blocks | 1.59 | 4.90 | multi-head attention + 512 context changes everything |
| 5b | gpt-neo style, 4 blocks, 3024 ff | 1.36 | 3.88 | close to the TinyStories paper |
| 6-7 | 5b + DPO | n/a | n/a | preferred by the judge 56.2% of the time (p = 0.0063) |

## caveats

things i know are not perfect:

- **the lstm and transformer losses use different tokenisers** (`vuiseng9/bpe-10.0k-tinystories` vs the retrained GPT-Neo one) and different context lengths (128 vs 512). both vocabularies are 10k and both give ~465m training tokens, so the numbers are roughly comparable, but it is not a controlled comparison.
- **the validation split doubles as the test set**, and it is also where the DPO training and evaluation prompts come from. the 500 evaluation prompts can overlap with the 2,500 training prompts (a lot of prompts are just "Once upon a time, there was a little girl named Lily."), although the completions are freshly sampled.
- **the judge is one small model.** in the evaluation, the original model's completion is always labelled `Completion 0` and the DPO model's `Completion 1`. the order is shuffled, but the labels are not. during data generation the judge chose id 0 a bit more often than chance, so if anything a label bias would work *against* the DPO model.
- **each evaluation prompt gets one sample per model** at temperature 1, so some of the 56.2% is sampling noise; this is what the confidence interval is for.

## running it yourself

the notebooks are not a packaged pipeline, and some paths are hardcoded (sorry).

- **data**: the TinyStories parquet files (train and validation splits) from hugging face.
- **python packages**: `torch`, `transformers`, `numpy`, `pandas` (with a parquet engine), `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, plus `ollama` and `pydantic` for the judge.
- **judge**: a running [ollama](https://ollama.com) server with `qwen3.5:4b` pulled.
- **weights**: tracked with git LFS, so run `git lfs pull` after cloning.

to load the best model and get a story out of it:

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
