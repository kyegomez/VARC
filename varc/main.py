import random
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Canvas:
    """
    Handles canvas operations for ARC tasks.

    The Canvas class provides functionality to place ARC grids on a fixed-size canvas
    with scaling and translation operations. This enables the model to process variable-sized
    ARC grids as fixed-size images, which is essential for vision-based processing.

    Attributes:
        canvas_size (int): Size of the square canvas (default: 64).
        num_colors (int): Number of color tokens in the ARC task (default: 10).
        bg_token (int): Token ID for background pixels.
        border_token (int): Token ID for border markers.
        total_tokens (int): Total number of token types (colors + background + border).

    Example:
        >>> canvas = Canvas(canvas_size=64, num_colors=10)
        >>> grid = torch.randint(0, 10, (5, 5))
        >>> placed = canvas.place_on_canvas(grid, scale=2, offset=(10, 10))
    """

    def __init__(self, canvas_size: int = 64, num_colors: int = 10):
        self.canvas_size = canvas_size
        self.num_colors = num_colors
        self.bg_token = num_colors  # Background token
        self.border_token = num_colors + 1  # Border token
        self.total_tokens = num_colors + 2

    def place_on_canvas(
        self,
        grid: torch.Tensor,
        scale: int = 1,
        offset: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Place a grid on canvas with scaling and translation.

        Args:
            grid: Input grid of shape (H, W)
            scale: Integer scaling factor
            offset: (y, x) offset for translation, random if None

        Returns:
            Canvas tensor of shape (canvas_size, canvas_size)
        """
        H, W = grid.shape

        # Scale the grid (nearest neighbor)
        if scale > 1:
            scaled_grid = grid.repeat_interleave(scale, dim=0).repeat_interleave(
                scale, dim=1
            )
        else:
            scaled_grid = grid

        H_scaled, W_scaled = scaled_grid.shape

        # Random offset if not provided
        if offset is None:
            max_y = self.canvas_size - H_scaled
            max_x = self.canvas_size - W_scaled
            offset_y = random.randint(0, max(0, max_y))
            offset_x = random.randint(0, max(0, max_x))
        else:
            offset_y, offset_x = offset

        # Create canvas filled with background token
        canvas = torch.full(
            (self.canvas_size, self.canvas_size),
            self.bg_token,
            dtype=grid.dtype,
            device=grid.device,
        )

        # Place scaled grid on canvas
        canvas[
            offset_y : offset_y + H_scaled,
            offset_x : offset_x + W_scaled,
        ] = scaled_grid

        return canvas

    def add_border_tokens(
        self,
        canvas: torch.Tensor,
        grid_shape: Tuple[int, int],
        scale: int,
        offset: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Add border tokens to indicate output shape boundaries.

        Border tokens are placed at the right and bottom edges of the placed grid
        to help the model understand the output dimensions during inference.

        Args:
            canvas: Canvas tensor of shape (canvas_size, canvas_size).
            grid_shape: Original grid shape (H, W) before scaling.
            scale: Scaling factor applied to the grid.
            offset: (y, x) offset where the grid was placed.

        Returns:
            Canvas tensor with border tokens added at the grid boundaries.
        """
        canvas = canvas.clone()
        H, W = grid_shape
        offset_y, offset_x = offset

        H_scaled = H * scale
        W_scaled = W * scale

        # Add border on right edge
        if offset_x + W_scaled < self.canvas_size:
            canvas[offset_y : offset_y + H_scaled, offset_x + W_scaled] = (
                self.border_token
            )

        # Add border on bottom edge
        if offset_y + H_scaled < self.canvas_size:
            canvas[offset_y + H_scaled, offset_x : offset_x + W_scaled] = (
                self.border_token
            )

        return canvas


class SeparablePositionalEmbedding2D(nn.Module):
    """
    2D separable positional embeddings for spatial awareness.

    This module generates 2D positional embeddings by separately learning embeddings
    for x and y coordinates, then combining them. This approach is more parameter-efficient
    than learning full 2D positional embeddings and works well for vision transformers.

    The embeddings are separable in the sense that x and y coordinates are embedded
    independently and then concatenated, allowing the model to learn spatial relationships
    along each axis separately.

    Attributes:
        d_model (int): Model dimension. Must be even for separable embeddings.
        pos_embed_y (nn.Parameter): Learnable positional embeddings for y-coordinates.
        pos_embed_x (nn.Parameter): Learnable positional embeddings for x-coordinates.

    Example:
        >>> pos_embed = SeparablePositionalEmbedding2D(d_model=512, max_len=32)
        >>> embeddings = pos_embed(h=16, w=16)  # Shape: (256, 512)
    """

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        self.d_model = d_model

        # Separate embeddings for x and y coordinates
        self.pos_embed_y = nn.Parameter(torch.randn(max_len, d_model // 2) * 0.02)
        self.pos_embed_x = nn.Parameter(torch.randn(max_len, d_model // 2) * 0.02)

    def forward(self, h: int, w: int) -> torch.Tensor:
        """
        Generate 2D positional embeddings.

        Args:
            h, w: Height and width in patches

        Returns:
            Positional embeddings of shape (h*w, d_model)
        """
        # Get y and x embeddings
        pos_y = self.pos_embed_y[:h]  # (h, d_model//2)
        pos_x = self.pos_embed_x[:w]  # (w, d_model//2)

        # Broadcast and concatenate
        pos_y = pos_y.unsqueeze(1).expand(h, w, -1)  # (h, w, d_model//2)
        pos_x = pos_x.unsqueeze(0).expand(h, w, -1)  # (h, w, d_model//2)

        pos_embed = torch.cat([pos_y, pos_x], dim=-1)  # (h, w, d_model)
        pos_embed = pos_embed.reshape(h * w, self.d_model)  # (h*w, d_model)

        return pos_embed


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) mechanism.

    GQA reduces memory and computation by sharing key-value heads across multiple
    query heads. This is more efficient than standard multi-head attention while
    maintaining similar representational capacity.

    In GQA, query heads are divided into groups, and each group shares a set of
    key-value heads. This reduces the number of key-value projections needed,
    making it more memory-efficient for large models.

    Attributes:
        d_model (int): Model dimension.
        n_query_heads (int): Number of query heads.
        n_kv_heads (int): Number of key-value heads (must divide n_query_heads).
        d_head (int): Dimension per head.
        head_scale (float): Scaling factor for attention scores.

    Example:
        >>> attn = GroupedQueryAttention(d_model=512, n_query_heads=8, n_kv_heads=2)
        >>> x = torch.randn(2, 100, 512)  # (batch, seq_len, d_model)
        >>> out = attn(x)  # Shape: (2, 100, 512)
    """

    def __init__(
        self,
        d_model: int,
        n_query_heads: int,
        n_kv_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert (
            d_model % n_query_heads == 0
        ), "d_model must be divisible by n_query_heads"
        assert (
            n_query_heads % n_kv_heads == 0
        ), "n_query_heads must be divisible by n_kv_heads"

        self.d_model = d_model
        self.n_query_heads = n_query_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_model // n_query_heads
        self.head_scale = self.d_head**-0.5
        self.n_groups = n_query_heads // n_kv_heads

        # Projections: separate for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of grouped query attention.

        Args:
            x: Input tensor of shape (B, N, d_model).
            attn_mask: Optional attention mask of shape (B, N, N) or (N, N).

        Returns:
            Output tensor of shape (B, N, d_model).
        """
        B, N, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # (B, N, d_model)
        k = self.k_proj(x)  # (B, N, n_kv_heads * d_head)
        v = self.v_proj(x)  # (B, N, n_kv_heads * d_head)

        # Reshape for multi-head attention
        q = q.view(B, N, self.n_query_heads, self.d_head).transpose(
            1, 2
        )  # (B, n_q_heads, N, d_head)
        k = k.view(B, N, self.n_kv_heads, self.d_head).transpose(
            1, 2
        )  # (B, n_kv_heads, N, d_head)
        v = v.view(B, N, self.n_kv_heads, self.d_head).transpose(
            1, 2
        )  # (B, n_kv_heads, N, d_head)

        # Repeat K and V for each query group
        k = k.repeat_interleave(self.n_groups, dim=1)  # (B, n_q_heads, N, d_head)
        v = v.repeat_interleave(self.n_groups, dim=1)  # (B, n_q_heads, N, d_head)

        # Compute attention scores
        attn_scores = (
            torch.matmul(q, k.transpose(-2, -1)) * self.head_scale
        )  # (B, n_q_heads, N, N)

        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)  # (B, 1, N, N)
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_out = torch.matmul(attn_weights, v)  # (B, n_q_heads, N, d_head)

        # Concatenate heads and project
        attn_out = (
            attn_out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        )  # (B, N, d_model)
        out = self.out_proj(attn_out)

        return out


class TransformerBlock(nn.Module):
    """
    Transformer block with Grouped Query Attention (GQA).

    A standard transformer block consisting of:
    1. RMS normalization
    2. Grouped Query Attention (more efficient than standard multi-head attention)
    3. Residual connection
    4. RMS normalization
    5. Feed-forward MLP
    6. Residual connection

    The use of GQA reduces memory and computation compared to standard multi-head attention
    while maintaining similar representational capacity. RMSNorm provides efficient normalization
    without mean centering, making it faster than LayerNorm while often achieving better performance.
    This is particularly beneficial for vision transformers processing high-resolution images.

    Attributes:
        norm1 (RMSNorm): RMS normalization before attention.
        attn (GroupedQueryAttention): Grouped query attention mechanism.
        norm2 (RMSNorm): RMS normalization before MLP.
        mlp (nn.Sequential): Two-layer MLP with GELU activation.

    Example:
        >>> block = TransformerBlock(d_model=512, n_query_heads=8, n_kv_heads=2, mlp_dim=2048)
        >>> x = torch.randn(2, 100, 512)
        >>> out = block(x)  # Shape: (2, 100, 512)
    """

    def __init__(
        self,
        d_model: int,
        n_query_heads: int,
        n_kv_heads: int,
        mlp_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = GroupedQueryAttention(
            d_model=d_model,
            n_query_heads=n_query_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
        )
        self.norm2 = nn.RMSNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer block.

        Args:
            x: Input tensor of shape (B, N, d_model).
            attn_mask: Optional attention mask of shape (B, N, N) or (N, N).

        Returns:
            Output tensor of shape (B, N, d_model).
        """
        # Self-attention with residual connection
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm, attn_mask)
        x = x + attn_out

        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))

        return x


class VisionARCModel(nn.Module):
    """
    Vision Transformer model for ARC (Abstraction and Reasoning Corpus) tasks.

    This model formulates ARC as an image-to-image translation problem using a Vision Transformer
    architecture. The model processes ARC grids by:
    1. Placing variable-sized grids on a fixed-size canvas
    2. Converting the canvas into patches
    3. Processing patches through transformer layers with grouped query attention
    4. Predicting output grids through per-pixel classification

    The model uses task-specific tokens for multi-task learning and supports test-time training
    for generalization to unseen tasks.

    Attributes:
        num_colors (int): Number of color tokens in ARC (default: 10).
        canvas_size (int): Size of the square canvas (default: 64).
        patch_size (int): Size of each patch (default: 2).
        d_model (int): Model dimension (default: 512).
        n_layers (int): Number of transformer layers (default: 10).
        n_query_heads (int): Number of query heads for attention (default: 8).
        n_kv_heads (int): Number of key-value heads for grouped query attention (default: 2).
        mlp_dim (int): Hidden dimension of MLP in transformer blocks (default: 512).
        dropout (float): Dropout rate (default: 0.1).
        num_tasks (int): Number of task-specific tokens (default: 400).

    Example:
        >>> model = VisionARCModel(
        ...     num_colors=10,
        ...     canvas_size=64,
        ...     patch_size=2,
        ...     d_model=512,
        ...     n_layers=10,
        ...     n_query_heads=8,
        ...     n_kv_heads=2,
        ...     mlp_dim=512,
        ... )
        >>> canvas = torch.randint(0, 10, (2, 64, 64))
        >>> logits = model(canvas, task_id=0)  # Shape: (2, 64, 64, 12)
    """

    def __init__(
        self,
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
    ):
        super().__init__()

        self.num_colors = num_colors
        self.canvas_size = canvas_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.n_patches = (canvas_size // patch_size) ** 2
        self.grid_size = canvas_size // patch_size

        # Canvas handler
        self.canvas = Canvas(canvas_size, num_colors)

        # Color embedding (for discrete color indices)
        self.color_embedding = nn.Embedding(
            num_colors + 2, d_model
        )  # +2 for BG and Border

        # Patch embedding: project patch to d_model
        self.patch_proj = nn.Linear(patch_size * patch_size * d_model, d_model)

        # Task conditional tokens
        self.task_tokens = nn.Parameter(torch.randn(num_tasks, d_model) * 0.02)

        # Positional embeddings
        self.pos_embed = SeparablePositionalEmbedding2D(d_model, max_len=self.grid_size)

        # Transformer blocks with grouped query attention
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_query_heads=n_query_heads,
                    n_kv_heads=n_kv_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)

        # Output projection: per-pixel classification
        self.output_proj = nn.Linear(d_model, num_colors + 2)  # Predict all token types

        self.dropout = nn.Dropout(dropout)

    def patchify(self, canvas: torch.Tensor) -> torch.Tensor:
        """
        Convert canvas to patches for transformer processing.

        This method divides the canvas into non-overlapping patches, embeds each pixel
        using color embeddings, and flattens patches for transformer input.

        Args:
            canvas: Input canvas tensor of shape (B, H, W) with discrete color indices
                where H = W = canvas_size.

        Returns:
            patches: Flattened patch embeddings of shape (B, n_patches, patch_size^2 * d_model)
                where n_patches = (canvas_size // patch_size)^2.
        """
        B, H, W = canvas.shape
        p = self.patch_size

        # Embed colors
        canvas_embed = self.color_embedding(canvas)  # (B, H, W, d_model)

        # Reshape to patches
        canvas_embed = canvas_embed.reshape(B, H // p, p, W // p, p, self.d_model)
        canvas_embed = canvas_embed.permute(
            0, 1, 3, 2, 4, 5
        )  # (B, H//p, W//p, p, p, d_model)
        patches = canvas_embed.reshape(B, (H // p) * (W // p), p * p * self.d_model)

        return patches

    def forward(
        self,
        canvas: torch.Tensor,
        task_id: int,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through the Vision ARC model.

        Processes the input canvas through patchification, positional encoding,
        task conditioning, transformer layers, and output projection to produce
        per-pixel color predictions.

        Args:
            canvas: Input canvas tensor of shape (B, canvas_size, canvas_size)
                with discrete color indices (0 to num_colors-1).
            task_id: Task identifier index (0 to num_tasks-1) for task-specific conditioning.
            return_attention: Whether to return attention maps (currently not implemented).

        Returns:
            logits: Output logits tensor of shape (B, canvas_size, canvas_size, num_colors+2)
                where the last dimension contains logits for each color class plus
                background and border tokens.
        """
        B = canvas.shape[0]

        # Patchify
        patches = self.patchify(canvas)  # (B, n_patches, patch_size^2 * d_model)

        # Project patches
        x = self.patch_proj(patches)  # (B, n_patches, d_model)

        # Add positional embeddings
        pos_embed = self.pos_embed(
            self.grid_size, self.grid_size
        )  # (n_patches, d_model)
        x = x + pos_embed.unsqueeze(0)

        # Add task token
        task_token = (
            self.task_tokens[task_id].unsqueeze(0).unsqueeze(0)
        )  # (1, 1, d_model)
        x = x + task_token

        x = self.dropout(x)

        # Create attention mask to ignore background tokens
        # This is optional but helps focus on foreground
        attn_mask = None  # Can be implemented based on canvas content

        # Transformer blocks
        for block in self.blocks:
            x = block(x, attn_mask)

        x = self.norm(x)

        # Output projection
        logits = self.output_proj(x)  # (B, n_patches, num_colors+2)

        # Reshape to canvas
        logits = logits.reshape(B, self.grid_size, self.grid_size, -1)

        # Upsample to original canvas size (each patch predicts patch_size^2 pixels)
        logits = logits.permute(0, 3, 1, 2)  # (B, C, H', W')
        logits = F.interpolate(
            logits,
            size=(self.canvas_size, self.canvas_size),
            mode="nearest",
        )
        logits = logits.permute(0, 2, 3, 1)  # (B, H, W, C)

        return logits

    def get_new_task_embedding(self) -> nn.Parameter:
        """
        Create a new randomly initialized task embedding for test-time training.

        This method is used when encountering a new task at test time. The embedding
        can be fine-tuned on a few examples from the new task to adapt the model.

        Returns:
            A new learnable task embedding parameter of shape (1, d_model).
        """
        return nn.Parameter(torch.randn(1, self.d_model) * 0.02)


class ARCDataAugmentation:
    """
    Data augmentation utilities for ARC tasks.

    This class provides geometric and color-based augmentation methods to increase
    the diversity of training data. All augmentations preserve the semantic structure
    of ARC tasks while applying transformations that are common in visual reasoning.

    The augmentations include:
    - Geometric: horizontal/vertical flips, rotations (90°, 180°, 270°)
    - Color: color permutation (preserves structure, changes appearance)

    All methods are static and operate on torch tensors representing ARC grids.

    Example:
        >>> grid = torch.randint(0, 10, (5, 5))
        >>> flipped = ARCDataAugmentation.flip_horizontal(grid)
        >>> rotated = ARCDataAugmentation.rotate_90(grid)
    """

    @staticmethod
    def flip_horizontal(grid: torch.Tensor) -> torch.Tensor:
        """
        Flip grid horizontally (left-right).

        Args:
            grid: Input grid tensor of shape (..., H, W).

        Returns:
            Horizontally flipped grid of the same shape.
        """
        return torch.flip(grid, dims=[-1])

    @staticmethod
    def flip_vertical(grid: torch.Tensor) -> torch.Tensor:
        """
        Flip grid vertically (top-bottom).

        Args:
            grid: Input grid tensor of shape (..., H, W).

        Returns:
            Vertically flipped grid of the same shape.
        """
        return torch.flip(grid, dims=[-2])

    @staticmethod
    def rotate_90(grid: torch.Tensor) -> torch.Tensor:
        """
        Rotate grid 90 degrees clockwise.

        Args:
            grid: Input grid tensor of shape (..., H, W).

        Returns:
            Rotated grid of shape (..., W, H).
        """
        return torch.rot90(grid, k=1, dims=[-2, -1])

    @staticmethod
    def rotate_180(grid: torch.Tensor) -> torch.Tensor:
        """
        Rotate grid 180 degrees.

        Args:
            grid: Input grid tensor of shape (..., H, W).

        Returns:
            Rotated grid of the same shape.
        """
        return torch.rot90(grid, k=2, dims=[-2, -1])

    @staticmethod
    def rotate_270(grid: torch.Tensor) -> torch.Tensor:
        """
        Rotate grid 270 degrees clockwise (or 90 degrees counter-clockwise).

        Args:
            grid: Input grid tensor of shape (..., H, W).

        Returns:
            Rotated grid of shape (..., W, H).
        """
        return torch.rot90(grid, k=3, dims=[-2, -1])

    @staticmethod
    def permute_colors(grid: torch.Tensor, permutation: List[int]) -> torch.Tensor:
        """
        Apply color permutation to the grid.

        This preserves the structure of the grid while changing the color mapping.
        Useful for augmentation as it maintains spatial relationships.

        Args:
            grid: Input grid tensor with color indices.
            permutation: List of new color indices, where permutation[i] is the new
                color for pixels that were originally color i.

        Returns:
            Grid with permuted colors of the same shape.
        """
        result = grid.clone()
        for old_color, new_color in enumerate(permutation):
            result[grid == old_color] = new_color
        return result

    @staticmethod
    def get_augmentations():
        """
        Get all standard geometric augmentation functions.

        Returns a list of (name, function) tuples where each function takes
        (input_grid, target_grid) and returns (augmented_input, augmented_target).
        This ensures input-output pairs are augmented consistently.

        Returns:
            List of (name, augmentation_function) tuples. The augmentation functions
            take (x, y) as input and return (augmented_x, augmented_y).
        """
        return [
            ("identity", lambda x, y: (x, y)),
            (
                "flip_h",
                lambda x, y: (
                    ARCDataAugmentation.flip_horizontal(x),
                    ARCDataAugmentation.flip_horizontal(y),
                ),
            ),
            (
                "flip_v",
                lambda x, y: (
                    ARCDataAugmentation.flip_vertical(x),
                    ARCDataAugmentation.flip_vertical(y),
                ),
            ),
            (
                "rot_90",
                lambda x, y: (
                    ARCDataAugmentation.rotate_90(x),
                    ARCDataAugmentation.rotate_90(y),
                ),
            ),
            (
                "rot_180",
                lambda x, y: (
                    ARCDataAugmentation.rotate_180(x),
                    ARCDataAugmentation.rotate_180(y),
                ),
            ),
            (
                "rot_270",
                lambda x, y: (
                    ARCDataAugmentation.rotate_270(x),
                    ARCDataAugmentation.rotate_270(y),
                ),
            ),
        ]


class ARCLoss(nn.Module):
    """
    Cross-entropy loss for ARC tasks with background masking.

    This loss function computes per-pixel cross-entropy loss while ignoring
    background pixels. This is important because the canvas representation
    includes background padding, and we only want to optimize the foreground
    pixels that contain actual ARC grid content.

    The loss is computed as the mean cross-entropy over all non-background pixels,
    ensuring that background regions don't contribute to the gradient.

    Attributes:
        bg_token (int): Token ID for background pixels (default: 10).

    Example:
        >>> loss_fn = ARCLoss(bg_token=10)
        >>> logits = torch.randn(2, 64, 64, 12)  # (B, H, W, num_classes)
        >>> targets = torch.randint(0, 10, (2, 64, 64))
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(self, bg_token: int = 10):
        super().__init__()
        self.bg_token = bg_token

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute per-pixel cross-entropy loss, ignoring background pixels.

        Args:
            logits: Model output logits of shape (B, H, W, C) where C is the number
                of classes (num_colors + 2 for background and border).
            targets: Ground truth labels of shape (B, H, W) with discrete color indices.

        Returns:
            Scalar loss value computed as the mean cross-entropy over non-background pixels.
        """
        B, H, W, C = logits.shape

        # Reshape for cross-entropy
        logits = logits.reshape(B * H * W, C)
        targets = targets.reshape(B * H * W)

        # Create mask to ignore background
        mask = (targets != self.bg_token).float()

        # Compute loss
        loss = F.cross_entropy(logits, targets, reduction="none")
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)

        return loss
