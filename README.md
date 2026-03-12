
# Quantum-inspired Trit Neural Network (QITNN)

QITNN is a **quantum-inspired neural network architecture** implemented on top of **Serenade**, the author's custom DSL/compiler/runtime stack. The model runs on standard GPUs and does **not** require quantum hardware. The core idea is to represent each learnable projection with a **ternary superposition of amplitudes** and to propagate the resulting state through the network as a **full 2D simplex state**, rather than collapsing it to a single scalar observable

In the current public implementation, the network is a **byte-level autoregressive Transformer-style model** with QTS (Quantum Ternary Superposition) projections in attention and feed-forward layers.

---

## Based on Serenade Language

Official Document: [SERENADE_DOCS](https://github.com/kaifczxc-lab/Serenade-Language/blob/SiritoriProjects/Docs.md)

---

## What “quantum-inspired trit” means here

A standard linear layer stores one scalar weight per connection. QITNN stores **three amplitudes per connection**:

- `a_neg` for the `|-1>` branch
- `a_zero` for the `|0>` branch
- `a_pos` for the `|+1>` branch

This defines a qutrit-like state:

```math
\psi = a_{-}\,|-1\rangle + a_{0}\,|0\rangle + a_{+}\,|+1\rangle
```

These are **classical amplitudes computed on a GPU**, not physical quantum states.

---

## Core projection: 3 GEMMs + Born normalization

For an input matrix `X` and a QTS weight triplet `(A_neg, A_zero, A_pos)`, the projection computes three raw channels:

```math
C_{-} = X A_{-}
```
```math
C_{0} = X A_{0}
```
```math
C_{+} = X A_{+}
```

Then it applies a Born-style normalization elementwise:

```math
Z = C_{-}^{2} + C_{0}^{2} + C_{+}^{2}
```

```math
P_{-} = \frac{C_{-}^{2}}{Z}, \qquad
P_{0} = \frac{C_{0}^{2}}{Z}, \qquad
P_{+} = \frac{C_{+}^{2}}{Z}
```

This gives a valid ternary probability state for every output coordinate.

---

## The main architectural idea: full 2D simplex state

Earlier QTS variants collapsed the ternary state to a single scalar:

```math
E = P_{+} - P_{-}
```

That is lossy, because different ternary states can map to the same scalar. For example, the pure zero state and a balanced `|-1>/|+1>` mixture both give `0` on that axis.

QITNN instead keeps the **full 2D state** of the ternary distribution using a centered simplex basis:

```math
x = P_{+} - P_{-}
```

```math
y = \frac{2P_{0} - P_{-} - P_{+}}{\sqrt{3}} = \sqrt{3}\,P_{0} - \frac{1}{\sqrt{3}}
```

This is the key formula of the architecture.

The three pure states become the vertices of an equilateral triangle:

- `|-1>` -> `(-1, -1/sqrt(3))`
- `|0>`  -> `( 0,  2/sqrt(3))`
- `|+1>` -> `( 1, -1/sqrt(3))`

So the model does **not** propagate a single observable; it propagates a **point inside the ternary simplex**.

---

## Why the 2D simplex state matters

A ternary probability state `(P-, P0, P+)` has three numbers but one constraint:

```math
P_{-} + P_{0} + P_{+} = 1
```

So it has exactly **two degrees of freedom**. That means:

- **1D** is incomplete and loses information.
- **2D** is the minimal complete representation.
- **3D** is redundant.

This is why QITNN uses a **true 2D internal state** for Q/K/V, attention outputs, hidden activations, and FFN activations.

---

## High-level architecture

QITNN is a Transformer-style autoregressive language model with:

- byte vocabulary (`VOCAB = 256`)
- token embeddings + positional embeddings
- pre-norm residual blocks
- QTS-based Q/K/V/O projections
- full **2D causal attention**
- QTS-based FFN projections
- standard output projection to vocabulary logits

In the current script:

- `DIM` is the logical feature width
- `HDIM = 2 * DIM` is the packed visible width storing `[x | y]`
- `FFN` is the logical FFN width
- `HFFN = 2 * FFN` is the packed FFN width storing `[x | y]`

The hidden state is always packed as:

```text
[x | y]
```

where the first half is the simplex x-axis and the second half is the simplex y-axis.

---

## End-to-end execution order

### 1. Tokenization / input window
A byte sequence is loaded into `tokens`, and the next-byte targets are loaded into `targets`.

### 2. Embedding stage
- `embed_lookup` fetches token embeddings.
- `pos_add` adds positional embeddings.

The result is a packed hidden tensor of shape `[SEQ, HDIM]`.

### 3. LayerNorm before attention
The hidden tensor is normalized with `layernorm_batch`.

### 4. QTS Q/K/V projection
Each of `Q`, `K`, and `V` is produced by `qts_forward3`:

- three raw GEMMs
- Born normalization
- output of two channels:
  - `u = P+ - P-`
  - `v = P0`

Then the second channel is recentered into the simplex basis:

```math
y = \sqrt{3} v - \frac{1}{\sqrt{3}}
```

So `Q`, `K`, `V` become full 2D simplex signals.

### 5. 2D causal attention
`causal_attention2` computes scores using **both** simplex axes:

```math
score_{ij} = \langle Q_x(i), K_x(j) \rangle + \langle Q_y(i), K_y(j) \rangle
```

with causal masking and softmax.

The output is also two-channel:

- `attn_out_x`
- `attn_out_y`

These are packed back into `[x | y]`.

### 6. QTS output projection (`wo`)
The packed attention output goes through another `qts_forward3`, producing a new 2D simplex state.

### 7. Residual connection
The projected attention output is added back to the residual stream.

### 8. LayerNorm before FFN
Another pre-norm step is applied.

### 9. QTS FF1 projection
The hidden state goes through `qts_forward3` into FFN width.

### 10. Nonlinearity
The x-channel goes through `GELU`, while the second channel is recentered and kept as the y-axis.

So FFN mid-state is:

```text
[GELU(x) | y]
```

### 11. QTS FF2 projection
The FFN mid-state is projected back to the hidden width using another QTS projection.

### 12. Residual connection
The FFN output is added back to the block input.

### 13. Final LayerNorm and output head
After the last block:

- final LayerNorm
- standard dense output projection
- byte logits

### 14. Loss
`celoss` computes cross-entropy and writes the gradient into the logits buffer in-place.

---

## Backpropagation order

The backward pass mirrors the forward pass.

### 1. Output head backward
`backward` computes gradients for the final dense projection.

### 2. Final LayerNorm backward
`layernorm_backward` propagates through the final normalization.

### 3. QTS backward through FF2 / FF1 / O / Q / K / V
Every QTS projection uses `qts_backnorm3` to backpropagate through the Born-normalized 2D head.

If the visible y-channel is

```math
y = \sqrt{3} P_0 - \frac{1}{\sqrt{3}}
```

then upstream gradients are converted back to the raw `P0` channel by:

```math
\frac{\partial L}{\partial P_0} = \sqrt{3} \frac{\partial L}{\partial y}
```

This is why the implementation multiplies y-side gradients by `sqrt(3)` before calling `qts_backnorm3`.

### 4. Standard linear backward per amplitude branch
After `qts_backnorm3`, gradients are split into three raw channels:

- `dc_neg`
- `dc_zero`
- `dc_pos`

Each branch is then backpropagated with the standard dense `backward` operator against:

- `A_neg`
- `A_zero`
- `A_pos`

### 5. Attention backward
`attention_backward2` propagates gradients through the full 2D attention operator:

- `dQx`, `dQy`
- `dKx`, `dKy`
- `dVx`, `dVy`

### 6. Embedding update
`embed_backward` updates the token embedding table for the tokens that appeared in the sequence window.

### 7. QTS stability step
`qts_prior` applies a small entropy-floor correction to each qutrit weight triplet if it starts collapsing too strongly.

---

## Stability mechanisms

### 1. Entropy regularization in `qts_backnorm3`
The backward kernel can include an entropy term to discourage hard collapse of ternary probabilities.

### 2. `qts_prior`
A soft entropy-floor prior is applied directly to weight triplets.
It is **silent above the entropy floor** and only activates when a triplet becomes too close to binary collapse.

### 3. `ZERO_BOOST`
The zero branch can be trained with a higher learning rate multiplier.
This compensates for optimization imbalance and helps the zero-state remain active instead of becoming a passive or dead channel.

---

## Operator reference (from the Serenade parser/runtime)

Below is the meaning of the GPU operators used by the model.

### Core QTS operators

#### `gpu qts_randinit Aneg Azero Apos count seed`
Initializes qutrit amplitudes with random values and L2-normalizes each triplet.

#### `gpu qts_forward E cn cz cp X Aneg Azero Apos M inDim outDim`
Legacy QTS forward head.
Performs 3 GEMMs and Born normalization, then outputs a **single scalar expected-value channel**.

#### `gpu qts_backnorm dcn dcz dcp dE cn cz cp count`
Backward pass for the legacy scalar QTS head.

#### `gpu qts_forward2 E P0 cn cz cp X Aneg Azero Apos M inDim outDim gate`
Intermediate dual-channel QTS head.
Outputs polarity plus a gated zero channel.

#### `gpu qts_backnorm2 dcn dcz dcp dE cn cz cp gate ent_lambda grad_gate count outDim`
Backward pass for the gated dual-channel QTS head.

#### `gpu qts_forward3 U V cn cz cp X Aneg Azero Apos M inDim outDim`
Main QTS forward operator used by the current architecture.
It performs:

- 3 linear projections
- Born normalization
- returns two separate channels:
  - `U = P+ - P-`
  - `V = P0`

#### `gpu qts_backnorm3 dcn dcz dcp dU dV cn cz cp ent_lambda count`
Main QTS backward operator for the current architecture.
Backpropagates through the two-channel Born-normalized simplex head.

#### `gpu qts_decay Aneg Azero Apos count factor`
Scales all three amplitude buffers by a factor.

#### `gpu qts_renorm Aneg Azero Apos count`
Renormalizes amplitude triplets to unit norm.

#### `gpu qts_prior Aneg Azero Apos step [entropyFloor] count`
Applies one small entropy-floor stabilization step to qutrit weight triplets, then renormalizes.

#### `gpu qts_diag Aneg Azero Apos count`
Prints diagnostics for qutrit weights, including:

- mean ternary probabilities
- entropy
- effective states
- collapse rate
- amplitude statistics

### Standard neural network operators used by QITNN

#### `gpu forward out in weights bias M inDim outDim`
Standard dense affine forward pass.

#### `gpu backward input weights gradOut gradW gradBias gradInput M inDim outDim`
Standard dense affine backward pass.
Computes gradients for weights, bias, and input.

#### `gpu sgd weights grad lr n`
In-place SGD update.

#### `gpu embed_lookup out table tokens seqLen dim vocab`
Sequence embedding lookup.

#### `gpu embed_backward table tokens grad seqLen dim vocab lr`
Embedding update for tokens that appeared in the current sequence.

#### `gpu pos_add x posTable seqLen dim`
Adds positional embeddings to the hidden state.

#### `gpu layernorm_batch x gamma beta seqLen dim`
Applies LayerNorm independently to each sequence position.

#### `gpu layernorm_backward dx dgamma dbeta x gamma dout seqLen dim`
Backward pass for batch LayerNorm.

#### `gpu gelu x n`
In-place GELU activation.

#### `gpu gelu_backward grad pre_gelu n`
Backpropagates through GELU.

#### `gpu causal_attention2 Qx Qy Kx Ky Vx Vy Ox Oy seqLen dim`
Full 2D causal self-attention over simplex state channels.

#### `gpu attention_backward2 dQx dQy dKx dKy dVx dVy Qx Qy Kx Ky Vx Vy dOx dOy seqLen dim`
Backward pass for the 2D causal attention operator.

#### `gpu celoss logits targets seqLen vocab`
Cross-entropy loss.
Computes the loss and writes `softmax - onehot` gradient into the logits buffer in-place.

#### `gpu copy dst src count`
Copies a buffer. The runtime can choose CPU copy, GPU copy, or device-host synchronization depending on pinning.

#### `gpu pin ptr count`
Pins a host buffer to a persistent device allocation.
This is used to keep QTS weights and key raw buffers resident on the GPU.

#### `gpu norm2 buf count label`
Prints the L2 norm of a buffer for debugging and gradient health checks.

---

## What is actually novel here

The most important idea is **not** just “three branches instead of one.”
The key idea is:

1. compute ternary probability states with a Born-like projection,
2. **do not collapse them to a single scalar**, and
3. propagate them through the model as a **full centered 2D simplex state**.

This gives the network a richer internal representation than a single expected value and makes the zero-state a first-class direction rather than a special gate or neutral placeholder.

---

## Relationship to Serenade

QITNN is **based on Serenade**:

- the model is written in the Serenade DSL,
- the parser lowers `gpu ...` statements into concrete C++/CUDA runtime calls,
- the runtime implements both standard neural network primitives and custom QTS kernels.

So the project is best understood as:

- **Serenade** as the language/runtime foundation,
- **Quantum Trit Kernel** as the low-level QTS CUDA backend,
- **QITNN** as the neural architecture built on top of both.

---

## Important implementation note

The current public script declares `LAYERS = 4`, but the manually instantiated forward/backward path currently materializes **two explicit blocks** (`layer 0` and `layer 1`). The architecture itself is general, but the present script executes two layers until the layer stack is generalized in code.

---

## Short summary

QITNN is a quantum-inspired Transformer-style language model that replaces standard linear projections with ternary amplitude projections, applies Born-style normalization, and carries the resulting ternary state through the network as a **full centered 2D simplex state**.

In short:

```text
3 projections -> Born normalization -> centered simplex [x|y] -> full 2D attention/FFN
```

That is the essence of the architecture.

# DISCLAIMER

This is an experimental MVP (Minimum Viable Product) created to demonstrate a novel concept (Quantum Trit Neural Network architecture).

Development Context: This project was developed with the assistance of AI. The core architecture, mathematical formulas, debugging, and system integration are my own work.

Hardware Specificity: The CUDA kernels are heavily optimized for my specific hardware (NVIDIA GeForce RTX 3060 Ti). They may contain bugs, perform poorly, or not work at all on other GPU architectures or configurations.

No Guarantees: This code is provided "AS IS", without any warranty or guarantee of correctness, performance, or suitability for any purpose. It works on my machine. That's all I can promise.

Constructive criticism is welcome, but open hate towards the entire work is not welcome.

If you're interested in the concept, feel free to explore, adapt, and contribute!

---

## Development Log

You can follow the original development process and discussions here:

[Discord Devlog Thread GPU Mode](https://discord.com/channels/1189498204333543425/1466534042768904356/1476227907327098931)

Or find it on GPU Mode Server in channel: #from-scratch
