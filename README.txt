storyteller :: maxxing out my 5060 TI to tell really good bedtime stories

hi, this is a bit of a passion project after i got a new pc to try to really learn how LLMs work. of course, none of my networks are "large", in fact, they'd be considered very small networks.
either way, the mechanisms of the networks are similar to the mechanisms of the chatbots that are prevalent in modern day times, but scaled to many orders of magnitude of data.

project goals ::
- [X] train lstm model, with experimentation on different layer and neuron counts
- [X] train transformer model, show dominance in performance vs lstm
- [ ] post training
    - [ ] llm-as-a-judge post-training with preferred outputs
    - [ ] llm-as-a-judge post-training with metrics scoring (creativity, correctness, etc.)
- [ ] model optimisations, and transformer variants

models ::
    lstm (2 layer, 768 neurons)                         :: validation loss 2.92, perplexity 18.51   :: weights/20260921_lstm.pt
    transformer (2 blocks, 768 neurons feed forward, 8 heads)    :: validation loss 1.59, perplexity 4.90    :: weights/20260925_768_768_8_8_0_0_transformer.pt
    transformer (4 blocks, 3024 neurons feed forward, 8 heads)   :: validation loss 1.36, perplexity 3.88    :: weights/20260925_3024_4x_8_4x_0_4x_transformer.pt