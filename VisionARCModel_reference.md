# VisionARCModel API Reference

## Overview

`VisionARCModel` is a Vision Transformer-based model designed for solving ARC (Abstraction and Reasoning Corpus) tasks. It formulates ARC as an image-to-image translation problem, processing variable-sized ARC grids through a fixed-size canvas representation and predicting output grids via per-pixel classification.

## Class Definition

```python
class VisionARCModel(nn.Module)
```

## Description

The `VisionARCModel` processes ARC tasks by:

1. **Canvas Representation**: Placing variable-sized ARC grids on a fixed-size canvas (default 64×64) with scaling and translation operations
2. **Patchification**: Converting the canvas into non-overlapping patches for transformer processing
3. **Vision Transformer Processing**: Processing patches through multiple transformer layers with grouped query attention (GQA)
4. **Task Conditioning**: Using task-specific tokens for multi-task learning
5. **Output Prediction**: Generating per-pixel color predictions through classification

The model uses efficient architectural components including:
- **Grouped Query Attention (GQA)**: Reduces memory and computation by sharing key-value heads across query groups
- **Separable Positional Embeddings**: 2D separable positional embeddings for spatial awareness
- **RMSNorm**: Efficient normalization without mean centering

## Initialization

### Constructor

```python
VisionARCModel(
    num_colors: int = 10,
    canvas_size: int = 64,
    patch_size: int = 2,
    d_model: int = 512,
    n_layers: int = 10,
    n_query_heads: int = 8,
    n_kv_heads: int = 2,
    mlp_dim: int = 512,
    dropout: float = 0.1,
    num_tasks: int = 400,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_colors` | `int` | `10` | Number of color tokens in ARC tasks. ARC uses 10 distinct colors (0-9) for grid cells. |
| `canvas_size` | `int` | `64` | Size of the square canvas in pixels. Variable-sized ARC grids are placed on this fixed-size canvas. Must be divisible by `patch_size`. |
| `patch_size` | `int` | `2` | Size of each patch in pixels. The canvas is divided into non-overlapping patches of this size. Must divide `canvas_size` evenly. |
| `d_model` | `int` | `512` | Model dimension (embedding dimension). Controls the width of the transformer layers. Common values: 256, 512, 768, 1024. |
| `n_layers` | `int` | `10` | Number of transformer blocks in the model. More layers increase model capacity but also computation and memory. |
| `n_query_heads` | `int` | `8` | Number of query heads for grouped query attention. Must be divisible by `n_kv_heads`. |
| `n_kv_heads` | `int` | `2` | Number of key-value heads for grouped query attention. These are shared across query groups, reducing memory usage. Must divide `n_query_heads` evenly. |
| `mlp_dim` | `int` | `512` | Hidden dimension of the MLP (feed-forward network) in transformer blocks. Typically set to `d_model` or `4 * d_model`. |
| `dropout` | `float` | `0.1` | Dropout rate for regularization. Applied in attention and MLP layers. Range: [0.0, 1.0]. |
| `num_tasks` | `int` | `400` | Number of task-specific tokens for multi-task learning. Each task gets a learnable embedding token. |

### Constraints

- `canvas_size` must be divisible by `patch_size`
- `d_model` must be divisible by `n_query_heads`
- `n_query_heads` must be divisible by `n_kv_heads`
- `d_model` must be even (for separable positional embeddings)

### Example: Basic Initialization

```python
import torch
from varc.main import VisionARCModel

# Minimal configuration
model = VisionARCModel()

# Custom configuration
model = VisionARCModel(
    num_colors=10,
    canvas_size=64,
    patch_size=2,
    d_model=512,
    n_layers=10,
    n_query_heads=8,
    n_kv_heads=2,
    mlp_dim=512,
    dropout=0.1,
    num_tasks=400,
)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1e6:.2f}M")
```

## Attributes

### Public Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `num_colors` | `int` | Number of color tokens (read-only) |
| `canvas_size` | `int` | Canvas size in pixels (read-only) |
| `patch_size` | `int` | Patch size in pixels (read-only) |
| `d_model` | `int` | Model dimension (read-only) |
| `n_patches` | `int` | Total number of patches: `(canvas_size // patch_size)²` |
| `grid_size` | `int` | Grid size in patches: `canvas_size // patch_size` |

### Internal Components

| Component | Type | Description |
|-----------|------|-------------|
| `canvas` | `Canvas` | Canvas handler for grid placement operations |
| `color_embedding` | `nn.Embedding` | Embedding layer for color tokens (size: `num_colors + 2`) |
| `patch_proj` | `nn.Linear` | Linear projection for patch embeddings |
| `task_tokens` | `nn.Parameter` | Learnable task-specific tokens (shape: `(num_tasks, d_model)`) |
| `pos_embed` | `SeparablePositionalEmbedding2D` | 2D separable positional embeddings |
| `blocks` | `nn.ModuleList` | List of transformer blocks |
| `norm` | `nn.LayerNorm` | Final layer normalization |
| `output_proj` | `nn.Linear` | Output projection for per-pixel classification |
| `dropout` | `nn.Dropout` | Dropout layer |

## Methods

### `forward`

Forward pass through the Vision ARC model.

#### Signature

```python
def forward(
    self,
    canvas: torch.Tensor,
    task_id: int,
    return_attention: bool = False,
) -> torch.Tensor
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `canvas` | `torch.Tensor` | Input canvas tensor of shape `(B, canvas_size, canvas_size)` with discrete color indices (0 to `num_colors-1`). Background pixels should use `num_colors` (bg_token). |
| `task_id` | `int` | Task identifier index (0 to `num_tasks-1`) for task-specific conditioning. Used to select the corresponding task token. |
| `return_attention` | `bool` | Whether to return attention maps. Currently not implemented, reserved for future use. |

#### Returns

| Return | Type | Description |
|--------|------|-------------|
| `logits` | `torch.Tensor` | Output logits tensor of shape `(B, canvas_size, canvas_size, num_colors+2)`. The last dimension contains logits for each color class (0 to `num_colors-1`), background token (`num_colors`), and border token (`num_colors+1`). |

#### Forward Pass Pipeline

1. **Patchification**: Convert canvas to patches and embed colors
2. **Patch Projection**: Project flattened patches to model dimension
3. **Positional Encoding**: Add separable 2D positional embeddings
4. **Task Conditioning**: Add task-specific token embedding
5. **Transformer Processing**: Pass through `n_layers` transformer blocks
6. **Normalization**: Apply final layer normalization
7. **Output Projection**: Project to output classes
8. **Upsampling**: Upsample from patch grid to full canvas resolution

#### Example

```python
import torch
from varc.main import VisionARCModel

model = VisionARCModel(
    num_colors=10,
    canvas_size=64,
    patch_size=2,
    d_model=512,
    n_layers=10,
    n_query_heads=8,
    n_kv_heads=2,
)

# Create input canvas (batch_size=4)
batch_size = 4
canvas = torch.randint(0, 10, (batch_size, 64, 64))
task_id = 0

# Forward pass
logits = model(canvas, task_id)
print(f"Output shape: {logits.shape}")  # (4, 64, 64, 12)

# Get predictions
predictions = torch.argmax(logits, dim=-1)
print(f"Predictions shape: {predictions.shape}")  # (4, 64, 64)
```

### `patchify`

Convert canvas to patches for transformer processing.

#### Signature

```python
def patchify(self, canvas: torch.Tensor) -> torch.Tensor
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `canvas` | `torch.Tensor` | Input canvas tensor of shape `(B, H, W)` with discrete color indices where `H = W = canvas_size`. |

#### Returns

| Return | Type | Description |
|--------|------|-------------|
| `patches` | `torch.Tensor` | Flattened patch embeddings of shape `(B, n_patches, patch_size² * d_model)` where `n_patches = (canvas_size // patch_size)²`. |

#### Process

1. Embed each pixel using `color_embedding` → `(B, H, W, d_model)`
2. Reshape into patches → `(B, H//p, W//p, p, p, d_model)`
3. Flatten patches → `(B, n_patches, p² * d_model)`

#### Example

```python
import torch
from varc.main import VisionARCModel

model = VisionARCModel(canvas_size=64, patch_size=2, d_model=512)
canvas = torch.randint(0, 10, (2, 64, 64))

patches = model.patchify(canvas)
print(f"Patches shape: {patches.shape}")  # (2, 1024, 2048)
# n_patches = (64/2)² = 32² = 1024
# patch_dim = 2² * 512 = 4 * 512 = 2048
```

### `get_new_task_embedding`

Create a new randomly initialized task embedding for test-time training.

#### Signature

```python
def get_new_task_embedding(self) -> nn.Parameter
```

#### Returns

| Return | Type | Description |
|--------|------|-------------|
| `task_embedding` | `nn.Parameter` | A new learnable task embedding parameter of shape `(1, d_model)` initialized with small random values. |

#### Use Case

This method is used when encountering a new task at test time. The embedding can be fine-tuned on a few examples from the new task to adapt the model without retraining the entire network.

#### Example

```python
import torch
from varc.main import VisionARCModel

model = VisionARCModel(num_tasks=400, d_model=512)

# Get a new task embedding for an unseen task
new_task_embedding = model.get_new_task_embedding()
print(f"New task embedding shape: {new_task_embedding.shape}")  # (1, 512)

# Use it for test-time training
# (This would require modifying the forward method or creating a custom version)
```

## Architecture Details

### Model Architecture

```
Input Canvas (B, 64, 64)
    ↓
Color Embedding (B, 64, 64, d_model)
    ↓
Patchification (B, 1024, patch_size² * d_model)
    ↓
Patch Projection (B, 1024, d_model)
    ↓
+ Positional Embeddings
    ↓
+ Task Token
    ↓
Transformer Blocks × n_layers
    ├─ RMSNorm
    ├─ Grouped Query Attention
    ├─ Residual Connection
    ├─ RMSNorm
    ├─ MLP (GELU)
    └─ Residual Connection
    ↓
Layer Normalization
    ↓
Output Projection (B, 1024, num_colors+2)
    ↓
Reshape & Upsample (B, 64, 64, num_colors+2)
```

### Token Types

The model uses three types of tokens:

1. **Color Tokens** (0 to `num_colors-1`): Standard ARC color indices
2. **Background Token** (`num_colors`): Used for canvas padding
3. **Border Token** (`num_colors+1`): Used to mark grid boundaries

### Grouped Query Attention (GQA)

GQA reduces memory and computation by sharing key-value heads:

- **Query Heads**: `n_query_heads` (e.g., 8)
- **Key-Value Heads**: `n_kv_heads` (e.g., 2)
- **Groups**: `n_query_heads // n_kv_heads` (e.g., 4)

Each group of query heads shares the same key-value heads, reducing the number of key-value projections needed.

### Separable Positional Embeddings

The model uses 2D separable positional embeddings:

- Separate embeddings for x and y coordinates
- Each coordinate dimension gets `d_model // 2` dimensions
- More parameter-efficient than full 2D embeddings
- Works well for vision transformers

## Usage Examples

### Basic Training Loop

```python
import torch
import torch.nn as nn
from torch.optim import Adam
from varc.main import VisionARCModel, ARCLoss

# Initialize model
model = VisionARCModel(
    num_colors=10,
    canvas_size=64,
    patch_size=2,
    d_model=512,
    n_layers=10,
    n_query_heads=8,
    n_kv_heads=2,
    mlp_dim=512,
    dropout=0.1,
    num_tasks=400,
)

# Loss function
loss_fn = ARCLoss(bg_token=10)

# Optimizer
optimizer = Adam(model.parameters(), lr=1e-4)

# Training step
model.train()
canvas = torch.randint(0, 10, (4, 64, 64))  # Input canvas
target = torch.randint(0, 10, (4, 64, 64))  # Target output
task_id = 0

# Forward pass
logits = model(canvas, task_id)

# Compute loss
loss = loss_fn(logits, target)

# Backward pass
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Loss: {loss.item():.4f}")
```

### Multi-Task Training

```python
import torch
from varc.main import VisionARCModel, ARCLoss

model = VisionARCModel(num_tasks=400)
loss_fn = ARCLoss(bg_token=10)

# Process different tasks
for task_id in range(400):
    canvas = torch.randint(0, 10, (2, 64, 64))
    target = torch.randint(0, 10, (2, 64, 64))
    
    logits = model(canvas, task_id)
    loss = loss_fn(logits, target)
    
    # Training step...
```

### Inference

```python
import torch
from varc.main import VisionARCModel

model = VisionARCModel()
model.eval()  # Set to evaluation mode

with torch.no_grad():
    canvas = torch.randint(0, 10, (1, 64, 64))
    task_id = 0
    
    logits = model(canvas, task_id)
    predictions = torch.argmax(logits, dim=-1)
    
    print(f"Predictions shape: {predictions.shape}")  # (1, 64, 64)
```

### Using Canvas for Variable-Sized Grids

```python
import torch
from varc.main import VisionARCModel, Canvas

model = VisionARCModel(canvas_size=64, num_colors=10)
canvas_handler = Canvas(canvas_size=64, num_colors=10)

# Variable-sized ARC grid
arc_grid = torch.randint(0, 10, (5, 7))  # 5×7 grid

# Place on canvas with scaling
placed_canvas = canvas_handler.place_on_canvas(
    arc_grid,
    scale=2,
    offset=(10, 10)
)

# Add batch dimension and process
canvas_batch = placed_canvas.unsqueeze(0)  # (1, 64, 64)
logits = model(canvas_batch, task_id=0)
```

## Performance Considerations

### Memory Usage

- **Model Parameters**: Approximately `O(d_model² * n_layers)` parameters
- **Activation Memory**: `O(batch_size * canvas_size² * d_model)` for intermediate activations
- **GQA Efficiency**: Reduces attention memory by `n_kv_heads / n_query_heads` compared to standard multi-head attention

### Computational Complexity

- **Time Complexity**: `O(batch_size * n_patches² * d_model * n_layers)`
- **Patch Count**: `n_patches = (canvas_size / patch_size)²`
- **GQA Benefit**: Reduces attention computation by sharing key-value heads

### Recommendations

1. **Small Models**: `d_model=256`, `n_layers=6`, `n_query_heads=4`, `n_kv_heads=2`
2. **Medium Models**: `d_model=512`, `n_layers=10`, `n_query_heads=8`, `n_kv_heads=2` (default)
3. **Large Models**: `d_model=768`, `n_layers=12`, `n_query_heads=12`, `n_kv_heads=4`

## Notes

1. **Canvas Size**: The canvas size should be large enough to accommodate the largest ARC grids in your dataset. Default 64×64 works well for most ARC tasks.

2. **Patch Size**: Smaller patch sizes (e.g., 2) provide finer spatial resolution but increase the number of patches. Larger patch sizes (e.g., 4) reduce computation but may lose fine-grained details.

3. **Task Tokens**: The number of task tokens (`num_tasks`) should match or exceed the number of distinct tasks in your training set. For test-time training, use `get_new_task_embedding()`.

4. **Background Handling**: The model uses `num_colors` as the background token. Ensure your input canvas uses this value for padding regions.

5. **Device Placement**: The model automatically handles device placement based on input tensors. Use `.to(device)` to move the model to GPU.

## Related Classes

- **`Canvas`**: Handles canvas operations for ARC tasks
- **`ARCLoss`**: Cross-entropy loss with background masking
- **`GroupedQueryAttention`**: Efficient attention mechanism
- **`TransformerBlock`**: Transformer block with GQA
- **`SeparablePositionalEmbedding2D`**: 2D positional embeddings

## References

- **Paper**: "ARC Is a Vision Problem!" (arXiv:2511.14761)
- **Repository**: [VARC GitHub](https://github.com/lillian039/VARC)

## See Also

- `example.py`: Complete usage example
- `README.md`: Project overview and installation instructions

