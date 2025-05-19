---
title: Mastering Gradients, `zero_grad`, and Optimizers in PyTorch
date: 2023-11-07 00:00:00 +0800
categories: [ML, Deep Learning]
tags: [ML]
math: true
---

# Mastering Gradients, `zero_grad`, and Optimizers in PyTorch

> *A practical guide to what actually happens under the hood when you train a neural network in PyTorch—and how to take full control of it.*

---

## Table of Contents

1. [Autograd Recap](#autograd-recap)
2. [`loss.backward()` — What Really Happens](#lossbackward—what-really-happens)
3. [Why Gradients **Accumulate**](#why-gradients-accumulate)
4. [`optimizer.zero_grad()` vs `model.zero_grad()`](#optimizerzero_grad-vs-modelzero_grad)
5. [Setting Gradients to `None` for Speed](#setting-gradients-to-none-for-speed)
6. [The `optimizer.step()` Update](#the-optimizerstep-update)
7. [Gradient Accumulation for Large Effective Batch Sizes](#gradient-accumulation-for-large-effective-batch-sizes)
8. [Best‑Practice Training Loop Templates](#best‑practice-training-loop-templates)
9. [Common Pitfalls & Debugging Tips](#common-pitfalls--debugging-tips)
10. [Cheat Sheet](#cheat-sheet)

---

## Autograd Recap

PyTorch builds a **dynamic computation graph** as you execute tensor operations. If a tensor has `requires_grad=True`, every subsequent operation records a *Function* node in the graph. The graph is **directed and acyclic** until you call `loss.backward()`, which traverses it in reverse to compute gradients via automatic differentiation.

```python
x = torch.randn(32, 100, requires_grad=True)
w = torch.randn(100, 10, requires_grad=True)
output = x @ w          # graph grows one op: matmul
loss = output.pow(2).mean()
```

* **Leaf tensors** (`x`, `w`) store a `.grad` attribute where gradients accumulate.
* **Non‑leaf tensors** (intermediate results) typically don’t hold gradients unless you explicitly call `.retain_grad()`.

---

## `loss.backward()` — What Really Happens

1. **Gradient seed**: If the loss is a scalar, autograd seeds the backward pass with a gradient of 1 w\.r.t. the loss.
2. **Reverse traversal**: PyTorch walks the graph backward, calling each Function’s `backward()` to compute `∂output/∂input`.
3. **Accumulation**: For every leaf parameter `p`, the computed gradient is **added** to `p.grad`:

   ```python
   p.grad = (p.grad or 0) + dp
   ```
4. **No parameter update yet**: `backward()` only fills `.grad`; you still need `optimizer.step()` to change the weights.

---

## Why Gradients **Accumulate**

* **Flexibility**: Lets you combine gradients from multiple forward passes (e.g. gradient accumulation, multi‑task losses, TBPTT).
* **Historical context**: Mirrors classical deep‑learning frameworks (Theano, Torch 7) where you manually zeroed grads.

If you *don’t* clear `.grad` between mini‑batches, your parameter updates will be **incorrect** because each step will mix gradients from multiple batches.

---

## `optimizer.zero_grad()` vs `model.zero_grad()`

```python
optimizer.zero_grad()  # preferred
model.zero_grad()      # identical effect
```

Both iterate over parameters and set `p.grad` **to zero** (`torch.zeros_like`). Use **one or the other**, not both.

Under the hood, `optimizer.zero_grad()` simply calls `p.grad = p.grad.detach().zero_()` for every parameter in the optimizer’s param groups.

### When might they differ?

If you pass a **subset** of parameters to the optimizer (rare but possible), `model.zero_grad()` clears **all** parameters—including ones the optimizer doesn’t know about. Usually that’s fine, but stick to `optimizer.zero_grad()` for clarity.

---

## Setting Gradients to `None` for Speed

Clearing gradients by *zero‑ing* writes to every element, wasting bandwidth. PyTorch ≥1.7 lets you instead **delete** the tensor and let autograd recreate it next backward:

```python
optimizer.zero_grad(set_to_none=True)
```

* **Pros**: Saves a kernel launch and memory bandwidth.
* **Cons**: You must check `p.grad is not None` before using `.grad` (e.g. for gradient clipping).

Alternatively, manual loop:

```python
for p in model.parameters():
    p.grad = None
```

---

## The `optimizer.step()` Update

After fresh gradients sit in `.grad`, call:

```python
optimizer.step()
```

This iterates through param groups and updates each parameter using the chosen rule (SGD, Adam, etc.). For Adam:

```python
m = beta1 * m + (1-beta1) * grad
v = beta2 * v + (1-beta2) * grad**2
param -= lr * m / (sqrt(v) + eps)
```

Order **always** matters:

1. `zero_grad()`
2. forward pass
3. `loss.backward()`
4. `optimizer.step()`

---

## Gradient Accumulation for Large Effective Batch Sizes

When a full batch won’t fit in GPU RAM:

```python
acc_steps = 4            # accumulate 4 mini‑batches
optimizer.zero_grad()
for i, batch in enumerate(loader):
    loss = compute_loss(batch) / acc_steps
    loss.backward()
    if (i+1) % acc_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

* Divide the loss by `acc_steps` so the total gradient matches that of a real large batch.
* Clip gradients **after** accumulation but **before** `step()`.

---

## Best‑Practice Training Loop Templates

### Standard Training Loop

```python
model.train()
for inputs, targets in loader:
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

### AMP + Grad‑Accumulation (Mixed Precision)

```python
scaler = torch.cuda.amp.GradScaler()
optimizer.zero_grad(set_to_none=True)

for i, batch in enumerate(loader):
    with torch.cuda.amp.autocast():
        loss = compute_loss(batch) / acc_steps
    scaler.scale(loss).backward()

    if (i+1) % acc_steps == 0:
        scaler.unscale_(optimizer)           # for gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
```

---

## Common Pitfalls & Debugging Tips

| Symptom                       | Likely Cause                                               | Fix                                                |
| ----------------------------- | ---------------------------------------------------------- | -------------------------------------------------- |
| **Loss oscillates wildly**    | Forgot to zero grads → accumulating across batches         | Call `optimizer.zero_grad()` each iteration        |
| **`NoneType` grad error**     | Using `set_to_none=True` and later assuming `.grad` exists | Check `param.grad is not None`                     |
| **Slow training**             | Zeroing large gradients on CPU before transfer             | Move model to GPU **before** calling `zero_grad()` |
| **Out‑of‑memory on backward** | Large batch                                                | Use gradient accumulation or checkpointing         |

---

## Cheat Sheet

* `loss.backward()`: computes **and adds** gradients to `.grad`.
* `optimizer.zero_grad()`: clears `.grad` (zero‑fill or `None`).
* `optimizer.step()`: updates params using current `.grad`.
* Call **zero\_grad → forward → backward → step** every update unless intentionally accumulating.

---

> **Remember**: Clearing gradients isn’t a performance hack; it’s about correctness. Treat `.grad` as a scratchpad—scribble a fresh set of numbers there every time you call `backward()`, unless you *want* them to add up.

Happy training! 🎉
