---
title: Understanding Tensor Axes in PyTorch
date: 2025-05-01 00:00:00 +0800
categories: [python]
tags: [python,pytorch]
---

# Understanding Tensor Axes in PyTorch

Tensors in PyTorch are generalizations of matrices to *n*-dimensions. One of the keys to writing correct deep learning code is getting comfortable with **axes** (often called **dimensions**). This guide walks you through:

1. What an axis is
2. How axes map to `.shape`
3. Common conventions (batch, channel, sequence)
4. Mnemonics to remember axes
5. Examples & visualizations
6. Pro tips for debugging and exploration

---

## 1. What Is an Axis?

A tensor’s **axes** are the numbered dimensions that define its shape. If a tensor has shape `(D0, D1, D2, ..., Dn)`, then:

* **Axis 0** corresponds to size `D0`
* **Axis 1** corresponds to size `D1`
* ...
* **Axis n** corresponds to size `Dn`

The label you attach to each axis depends on context (e.g., `batch`, `rows`, `cols`, `channels`, `sequence`, `feature`).

---

## 2. Axis Numbering → Position in `.shape`

The simplest rule is:

> **Axis *k* = the (*k*+1)th entry in `tensor.shape`**

### 2D Example: Matrices

```python
import torch
mat = torch.tensor([
    [1, 2, 3],  # 3 columns
    [4, 5, 6]   # 2 rows
])  # shape = (2, 3)
print(mat.shape)         # → torch.Size([2, 3])
print(mat.sum(dim=0))    # sum down each column → shape (3,)
print(mat.sum(dim=1))    # sum across each row  → shape (2,)
```

* `dim=0` collapses rows (vertical reduction): each column is summed
* `dim=1` collapses columns (horizontal reduction): each row is summed

---

## 3. Common Conventions

| Context              | Typical Shape Order                | Semantic Axes          |
| -------------------- | ---------------------------------- | ---------------------- |
| Stacked matrices     | `(depth, rows, cols)`              | (axis 0 = depth slice) |
| Image batches        | `(batch, channels, height, width)` | axis 0 = batch         |
| Sequences of vectors | `(batch, seq_len, feature_dim)`    | axis 1 = time/sequence |

**Key:** Always read axes *left to right* in `shape`.

---

## 4. Mnemonics to Remember

1. **“0 is first, 1 is second, 2 is third.”**
2. **Vertical vs. Horizontal**

   * `dim=0` (collapse vertically) → sums over rows → yields one value per column
   * `dim=1` (collapse horizontally) → sums over columns → yields one value per row
3. **Batch/Seq/Feature** (`B, N, D`):

   * `x.shape = (B, N, D)` →

     * `dim=0` averages across the *batch*
     * `dim=1` pools over the *sequence*
     * `dim=2` reduces the *feature* dimension

---

## 5. Examples & Visualizations

### 5.1 Matrix (2D)

```
      Col0  Col1  Col2
Row0   1     2     3
Row1   4     5     6
```

`.shape = (2 rows, 3 cols)`

### 5.2 Stack of Matrices (3D)

A tensor with `shape = (2, 3, 4)` can be visualized as two 3×4 “pages”:

```
Depth 0:                Depth 1:
 ┌────────────┐         ┌────────────┐
 │ 0  1  2  3 │         │12 13 14 15 │
 │ 4  5  6  7 │   and   │16 17 18 19 │
 │ 8  9 10 11 │         │20 21 22 23 │
 └────────────┘         └────────────┘
```

* `x[0]` → first page (axis 0 index)
* `x[:,1,:]` → row 1 of *every* page → shape `(2, 4)`
* `x[:,:,2]` → col 2 of *every* page → shape `(2, 3)`

---

## 6. Pro Tips

1. **Print Shapes Inline**

   ```python
   y = model(x)
   print(x.shape, '→', y.shape)
   ```
2. **Toy Shapes**
   Test with small `B, N, D` like `(1,4,3)` to sanity-check operations.
3. **`einsum` Notation**
   Sometimes `torch.einsum('b i d, b j d -> b i j', Q, K)` is more descriptive than `matmul`.
4. **Negative Dims**
   `dim=-1` always refers to the last axis; `dim=-2` the second-to-last.

---

With these rules—**axis numbering = position in `.shape`**, context-based labels, and shape-printing—you’ll read and write PyTorch tensor code with confidence. Happy coding!
