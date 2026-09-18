# micrograd, vectorized — from scratch

A reverse-mode autodiff engine where each node holds a whole **array** instead of a single
number, plus the dense layers built on top of it. Runs on the CPU through numpy, or on an
NVIDIA GPU through cupy, by changing one import.

This is the follow-up to [micrograd-from-scratch](https://github.com/DjErok/micrograd-from-scratch),
which is the scalar version. Same chain rule, same topological sort, same `_backward`
closures — but one `Tensor` replaces hundreds of thousands of `Value` objects.

## Attribution

**Every line of `engine.py` was typed by hand** after working through Andrej Karpathy's
micrograd videos, as a learning exercise. Nothing was generated or copied in.

`README.md` — this file — is the one exception. It is AI-written, because writing it wasn't
the point of the exercise.

## Why vectorize

For `MLP(784, [256, 10])` on one image, the scalar engine builds roughly **400,000** graph
nodes — one per multiply, one per add. Each is a Python object with a closure and a set.

The vectorized engine builds **10**, for any batch size:

```
x @ w      ->  1 node     (does all 200,704 multiply-accumulates inside one BLAS call)
+ b        ->  1 node
.tanh()    ->  1 node
```

Node count stops depending on how much data there is. It depends only on how many lines of
code you wrote. The arithmetic is identical; the Python interpreter overhead is what
disappears.

Measured, same architecture, one epoch of 60,000 images:

| | ms/step | notes |
|---|---|---|
| scalar engine, 1 image/step | ~88 | ~88 min per epoch |
| this engine, numpy, batch 1024 | 7.18 | 0.42 s per epoch |
| this engine, cupy, batch 1024 | 1.59 | 0.09 s per epoch |

## Usage

```python
from engine import Tensor, MLP, bench
import numpy as np

m = MLP(784, [256, 10])

X = Tensor(np.random.uniform(0, 1, (32, 784)))     # 32 images, 784 pixels
T = Tensor(np.random.uniform(-1, 1, (32, 10)))     # 32 targets, +/-1

loss = ((m(X) - T) ** 2).sum()

for p in m.parameters():
    p.grad = np.zeros_like(p.data)
loss.backward()

for p in m.parameters():
    p.data -= 0.01 * p.grad / 32
```

Built-in benchmark, which also serves as a full training loop and returns the trained model:

```python
m = bench(X, T, bnum=32, nin=784, hidden=[256], nout=10, epoch=20, actout=False)
```

`X` is `(n, features)` and `T` is `(n, outputs)`, both plain numpy. It reshapes `X` for you
but does not rescale it — normalize before calling. For MNIST:

```python
m = bench(images / 255, np.eye(10)[labels] * 2 - 1)

Xt = Tensor(test.reshape(len(test), 784) / 255)
pred = m(Xt).numpy().argmax(axis=1)
print((pred == test_labels).mean())
```

## Output activation

`Layer` takes `act=True`; `MLP` takes `afunc`, which decides whether the **final** layer gets
tanh:

```python
self.layers = [Layer(sz[i], sz[i+1], act=(i != last or afunc)) for i in range(len(nouts))]
```

Hidden layers always get tanh — without a nonlinearity between them, `x @ w1 @ w2` collapses
to a single linear layer and the hidden units buy nothing.

The output layer is a choice. tanh there caps outputs at `(-1, 1)`, and its derivative
`1 - tanh²` goes to zero exactly when a prediction is confidently wrong, so those cases
barely learn. `afunc=False` leaves the last layer linear, which removes that term from the
gradient. `argmax` is unaffected either way, since tanh is monotonic.

## CPU or GPU

One import decides where every array lives:

```python
try:
    import cupy as _cp
    xp = _cp
    ON_GPU = True
except Exception:
    xp = np
    ON_GPU = False
```

Everything below that line says `xp`. `_unbroadcast`, `__matmul__` and `backward()` are
identical either way, because cupy mirrors the numpy API deliberately.

Two things have no numpy equivalent and are guarded by `ON_GPU`:

- `sync()` — cupy queues kernels and returns immediately, so a timer without a barrier
  measures how fast you *queued*, not how fast it ran
- `.numpy()` — copies a device array back to host memory, for printing or plotting

`DTYPE` is `float32` on purpose. Consumer GeForce cards run fp64 at roughly 1/64 the fp32
rate, so leaving numpy's float64 default in place can make the GPU slower than the CPU.

```
pip install cupy-cuda12x[ctk]
```

Without cupy installed, it falls back to numpy and still works.

## What's in `engine.py`

| | |
|---|---|
| `Tensor` | array + gradient + graph edges + `_backward` closure |
| `_unbroadcast` | the part that doesn't exist in the scalar version (see below) |
| `__matmul__` | new op; backward is two more matmuls, via transposes |
| `Layer` | `w` is `(nin, nout)` — one **column** per neuron; `act` toggles tanh |
| `MLP` | stack of layers; `afunc` controls the output layer's activation |
| `bench` | batched training loop with timing, returns the trained model |

There is no `Neuron` class. A neuron is column `j` of `w`; 256 neurons is
`w.shape[1] == 256`. Architecture stops being Python class structure and becomes array
geometry — the same reason real frameworks have `nn.Linear(784, 256)` and no neuron object.

## The one genuinely new idea: un-broadcasting

Scalars can't broadcast, so the scalar engine never needed this.

A bias is `(256,)` but the batched output is `(1024, 256)`. numpy silently copies the bias
into all 1024 rows. That copy is 1024 *uses* of one parameter, and the chain rule says a
value used in many places accumulates the sum of its paths:

```
forward:   broadcast = copy
backward:  undo copy  = sum
```

```python
while g.ndim > len(shape):           # broadcasting added leading axes
    g = g.sum(axis=0)
for i, s in enumerate(shape):        # broadcasting stretched size-1 axes
    if s == 1 and g.shape[i] != 1:
        g = g.sum(axis=i, keepdims=True)
```

It is a no-op when the shapes already match, so `__add__` and `__mul__` call it
unconditionally. Without it, `b.grad += out.grad` raises
`ValueError: non-broadcastable output operand with shape (256,) doesn't match the broadcast
shape (1024, 256)` on the first backward pass.

`__matmul__` doesn't need it — matmul contracts the batch axis by itself.

## Verification

Gradients were checked against PyTorch on a 5→4→3 MLP with identical weights:

```
loss      mine  7.29691505   torch  7.29691410   diff 9.54e-07
w1.grad   mine -8.41627598   torch -8.41627598   diff 0.00e+00
b1.grad   mine -2.91527176   torch -2.91527200   diff 2.38e-07
w2.grad   mine -8.95719337   torch -8.95719242   diff 9.54e-07
b2.grad   mine 12.25293159   torch 12.25293255   diff 9.54e-07
```

Both activation modes match:

```
tanh out    loss 7.296915  torch 7.296914   max grad diff 9.54e-07
linear out  loss 7.299898  torch 7.299898   max grad diff 9.54e-07
```

Every difference is float32 rounding.

Speed against PyTorch on the same net (784→256→10, tanh, MSE, SGD), CPU,
forward + backward + update:

| batch | this engine | torch | ratio |
|---|---|---|---|
| 32 | 3.20 ms | 0.26 ms | 12.1x |
| 256 | 3.19 ms | 0.59 ms | 5.4x |
| 1024 | 9.30 ms | 1.81 ms | 5.1x |
| 4096 | 32.43 ms | 4.84 ms | 6.7x |

The remaining gap is memory, not algorithm: every intermediate here allocates a full
`zeros_like` gradient array that torch never creates, and `x @ w + b` is two kernels where
torch fuses one `addmm`.

## Requirements

```
numpy
cupy-cuda12x[ctk]    # optional, GPU only
```

## License

MIT
