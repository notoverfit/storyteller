## weight initialisations

in the past notebooks, we have discussed extensively about model details (lstms vs transformers) as well as post-training, using llm-as-a-judge DPO optimisation. what we have no explored in detail (which we should have) is the actual training process, and how researchers can track training issues well before the end-of-training. furthermore, the ever-important discussion of *weight initialisation* has not yet been discussed, which we aim to do here.

it is a fact that weight initialisation can make-or-break an experiement. for particularly difficult tasks, the wrong initialisation regime (just some random number, or 0) may lead to a network that does not converge, whereas optimal initialisation could lead to quick convergence. in the academia, there exists two main initialisation methods (for different activation functions):
1. Glorot initialisation (`sigmoid/tanh`): based on the number of input/output connections
2. He initialisation (`ReLU`): input count and a multiplier

as well, transformers are *residual networks* by nature, which we discuss the implications of briefly.

a long time ago, the frontier of deep learning was unable to make any progress due to gradient issues and the inability for problems to generalise. a principal reason for this inability to learn was *poor initialisation*. if you've ever tried to code up a neural net with just as vector library (like `numpy`), you may have experienced this yourself. even for simple non-linear problems like concentric circles, neural nets can quickly get stuck in local minima that do not improve.

### the set up of optimal initialisation

glorot and he initialisation are derived in the same way, but with different underlying assumptions. we define the pre-activated output at some layer $i$ and unit $k$ as
$$ s_k^i = \sum_l z_l^iW_{l, k}^i + b_k^i$$
where $z_k^{i} = f(s_k^{i-1})$, $f$ is the activation function and $b$ is the bias. the derivative of the cost function $C$ w.r.t $s_k$ can be defined with the forward layer (one-layer rule).

$$
\begin{align*}
    \frac{\partial C}{\partial s_k^i} = \sum_j \frac{\partial C}{\partial s_j^{i+1}} \frac{\partial s_j^{i+1}}{\partial s_k^i}
\end{align*}
$$

since $s_k^{i+1} = \sum_l f(s_k^i) W_{l, k}^{i+1} + b_k^{i+1}$, we can resolve the above into

$$
f'(s^i_k) \sum_j W_{k, j}^{i+1} \frac{\partial C}{\partial s_j^{i+1}}
$$

now the forward pass given the previous layer (one-layer rule) is
$$
s_k^{i} = \sum_l^{n_i} z_l^iW_{l, k}^i
$$

#### glorot initialisation

for glorot initialisation, we focus on hyperbolic tangent activation functions which act linearly near $0$. thus, $f(s) \approx s$ and $f'(s) \approx 1$. we also assume that weights within the same layer $i$ share the same variance $\text{Var}[W^i]$, and that the $E[W^i] = \mathbf{0}$. from this, we *aim to find the variance of the forward pass and the backward pass*.

a layer $i$ has $n_i$ units. thus, we find

$$
\begin{align*}
    \text{Var}[s^i_k] &= \sum_{l=1}^{n_i} \text{Var}[z^i_l W_{l, k}] \\
    &= \sum_{l=1}^{n_i} \text{Var}(z^i_l) \text{Var}(W^i_{l, k}) \\
    \text{Var}[s^i] &= n_i\text{Var}(W^i)\text{Var}(z^i)
\end{align*}
$$

the linear assumption holds that $z^i = f(s^{i-1}) = s^{i-1}$. we can then recursively take the variance definitions
$$
\text{Var}(s^i) = n_i\text{Var}(W^i)[n_{i-1}\text{Var}(W^{i-1})\text{Var}(z^{i-1})]
$$
and so on; leading to $\text{Var}(s^i) = \text{Var}(x) \prod_{i=1}^{i-1} n_i \text{Var}(W^i)$. so the variance of the output is given by the variance of the input and the product of all the weights prior. it should become slowly apparent why deep learning researchers talk about 'exploding' and 'vanishing' now - but it will become clearer in the back propagation.

we define the gradient of the cost function w.r.t $s^i_k$ as $g_i = \frac{\partial C}{\partial s^i_k}$. first note that:
- $s^i, z^{i+1}$ both have $n_{i+1}$ entries.
- $W^{i+1} \in \mathbb{R}^{i+1\times i+2}$

we found before that 

$$ g^i_k = f'(s^i_k) \sum_{j=1}^{n_{i+2}} W_{k, j}^{i+1}g_j^{i+1}$$

by the linear assumption, $f'(s^i_k) = 1$, thus simplifying to $\sum_{j=1}^{n_{i+2}} W_{k, j}^{i+1}g_j^{i+1}$. assume the terms are uncorrelated (they are). we can then find the variance of the gradient at layer $i$ as

$$ 
\begin{align*}
\text{Var}(g^i) &= n_{i+2}\text{Var}(W^{i+1})\text{Var}(g^{i+1}) \\
&= n_{i+2}\text{Var}(W^{i+1})(n_{i+3}\text{Var}(W^{i+2})\text{Var}(g^{i+2})) \\
&= \dots \\
&= \text{Var}(g^d) \prod_{l = i+1}^d n_{l + 1}\text{Var}(W^l)
\end{align*}
$$
where $d$ is the last layer. **our goal is for variance to be stable through the forward and backward pass** - this can clearly be achieved by making the multiples in each pass $n_i\text{Var}(W^i)$ and $n_{i+1}\text{Var}(W^i)$ be 1. thus
$$
\begin{align*}
    \text{Var}(W^i) &= \frac{1}{n_i} \\
    \text{Var}(W^i) &= \frac{1}{n_{i+1}}
\end{align*}
$$
of course this is only possibl ewhen $n_i = n_{i+1}$, which is very restrictive. xavier (glorot) instead allows for the reciprocal of variance to equal the average number of units across the two layers
$$
\begin{align*}
\frac{1}{\text{Var}(W^i)} &= \frac{n_i + n_{i+1}}{2} \\
\text{Var}(W^i) &= \frac{2}{n_i + n_{i+1}}
\end{align*}
$$
since xavier (glorot) uniformly samples the weights from some range $[-a, a]$, the variance of that distribution is given by
$$
\text{Var}(w) = \mathbb{E}(X^2) = \int_{-a}^a \frac{w^2}{2a}dw = \frac{a^2}{3}
$$
thus we find the "optimal" $a$
$$
\begin{align*}
\frac{a^2}{3} &= \frac{2}{n_i + n_{i+1}} \\
a &= \frac{\sqrt{6}}{\sqrt{n_i + n_{i+1}}}
\end{align*}
$$

#### he initialisation

whilst sigmoid and hyperbolic tangent activations dominated much of earlier deep learning research, rectified linear units (ReLU) and it's variants have mostly taken over for preferred activations. in a sigmoid $\sigma(x) = (1 + \exp(x))^{-1}$, the derivative w.r.t $x$ is $d_x\sigma(x) = \sigma(x) \cdot (1 - \sigma(x))$ which can become tiny at large $x$ or small $x$; that is, once activations become "saturated", the weights fail to learn. ReLU, defined by $f(x) = \max(0, x)$, and it's derivative defined as a piecewise function
$$
\frac{df(x)}{dx} = \begin{cases}
    1 & \text{if }x > 0 \\
    0 & \text{otherwise}
\end{cases}
$$

can continue to learn linearly when $x > 0$. however, many of our assumptions from the glorot initialisation no longer hold; the function does not have a linear regime when $x \approx 0$. previously in the forward pass, we computed $\text{Var}(zW)$, which we simplified easly as $\text{E}[z^2] = 0$ under hyperbolic tangent activations. in ReLU this is not the case. however $s^{i}$ is still symmetric for all $i \in \text{layers}$ and thus $s^{i}$ is symmetric with density $p$ and $\mathbb{E}(s^{i-1}) = 0$.
$$
\begin{align*}
    E[(z^i)^2] &= \int_{-\infty}^\infty s^2 p(s) ds \\
    &= \int_{0}^\infty s^2 p(s) ds
\end{align*}
$$
since $z = \max(0, s)$, the integrand when $s < 0$. now consider $\mathbb{E}[s^2] = \text{Var(s)}$ (since $\mathbb{E}(s) = 0$).

$$
\begin{align*}
    \mathbb{E}[s^2] &= \int_{-\infty}^\infty s^2 p(s) ds = \int_{-\infty}^0 s^2p(s) ds + \int_0^{\infty} s^2p(s)ds \\
\end{align*}
$$
now substituting $u = s$ and remember that $s^2$ is even and $p$ is symmetric
$$
A = \int_{\infty}^0 (-u)^2p(u) (-du) = \int_{0}^\infty u^2 p(u) du
$$

substituting $u$ back with $s$, we find that $\mathbb{E}[z^2] = \frac{1}{2}\text{Var}(s)$. now, walking through the forward pass, we have the following.
$$
\begin{align*}
\text{Var}(s_k^i) &= \sum_l \text{Var}(z_l^iW_{l, k}^i) \\
\text{Var}(zW) &= \mathbb{E}[z^2]\mathbb{E}[W^2] - (\mathbb{E}[z]E[W])^2 = \text{Var}(W)\mathbb{E}[z^2]  \\
\text{Var}(s^i) &= \frac{n_i}{2}\text{Var}(W^i)\text{Var}(s^{i-1})
\end{align*}
$$
this cascades into the first layer, where $\text{Var}(s^1) = n_1\text{Var}(W^1)\mathbb{E}[x]^2$.

$$ \text{Var}(s^i) = n_1 \text{Var}(W^1)\mathbb{E}[x^2]\sum_{l=1}^i \frac{n_l}{2}\text{Var}(W^l)$$

thus the forward pass condition is $\text{Var}(W^l) = \frac{2}{n_l}$. now, consider the backwards pass letting $g_k^i = \partial C / \partial s_k^i$, we can define the chain rule
$$ g_k^i = f'(s^i_k)\sum_{j=1}^{n_{i+2}} W_{k, j}^{i+1}g_j^{i+1}$$

we assume that the weights are independent of the gradient, and thus $\mathbb{E}[g_k^i] = 0$ and $\text{Var}(g^i) = \mathbb{E}[(g^i)^2]$. 

$$
\mathbb{E}[(g_k^i)^2] = \mathbb{E}[f'(s_k^i)^2]\mathbb{E}\left[\left(\sum_j W_{k, j}^{i+1} g_{j}^{i+1}\right)^2 \right]
$$

consider $\mathbb{E}[f'(s_k^i)^2]$. this is essentially a binary piecewise function. thus $\mathbb{E}[f'(s_k^i)^2] = \mathbb{E}[f'(s_k^i)]$. therefore, the expected value is $\frac{1}{2}$.
$$
\begin{align*}
\text{Var}(g^i) &= \frac{1}{2}n_{i+2}\text{Var}(W^{i+1})\text{Var}(g^{i+1}) \\
&= \text{Var}(g^d) \prod_{l = i+1}^d \frac{1}{2}n_{l+1}\text{Var}(W^l)
\end{align*}
$$
where $d$ is the last layer; and $n_{l+1}$ since $W_i$ is of dimension $n_{i} \times n_{i+1}$. therefore the backward constraint with unit variance is
$$
\text{Var}(W^l) = \frac{2}{n_{l+1}}
$$
**instead of asserting that these variances must equal to 1, he instead chooses the fan-in unit variance constraint and makes the fan-out variance equal to the constraint.** thus, we have

$$
\frac{1}{2}n_{l+1}\cdot \frac{2}{n_l} = \frac{n_{l+1}}{n_l}
$$

you will notice that as you expand out from $l \to d$ where $d$ is the final layer and $l \to s$ where $s$ is the first layer, we find this becomes $\frac{n_{d+1}}{n_{s+1}}$. we can thus just use the constraint $\frac{2}{n_l}$, to find the optimal uniformly sampled variance $\sqrt{\frac{6}{n_l}}$.

### where can this be applied?

the appropiate initialisations above are already set as defaults in pytorch, so no changes are necessary.