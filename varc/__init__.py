import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import random


class Canvas:
    """Handles canvas operations for ARC tasks."""

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
            scaled_grid = grid.repeat_interleave(
                scale, dim=0
            ).repeat_interleave(scale, dim=1)
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
        """Add border tokens to indicate output shape."""
        canvas = canvas.clone()
        H, W = grid_shape
        offset_y, offset_x = offset

        H_scaled = H * scale
        W_scaled = W * scale

        # Add border on right edge
        if offset_x + W_scaled < self.canvas_size:
            canvas[
                offset_y : offset_y + H_scaled, offset_x + W_scaled
            ] = self.border_token

        # Add border on bottom edge
        if offset_y + H_scaled < self.canvas_size:
            canvas[
                offset_y + H_scaled, offset_x : offset_x + W_scaled
            ] = self.border_token

        return canvas


class SeparablePositionalEmbedding2D(nn.Module):
    """2D separable positional embeddings."""

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        self.d_model = d_model

        # Separate embeddings for x and y coordinates
        self.pos_embed_y = nn.Parameter(
            torch.randn(max_len, d_model // 2) * 0.02
        )
        self.pos_embed_x = nn.Parameter(
            torch.randn(max_len, d_model // 2) * 0.02
        )

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
        pos_y = pos_y.unsqueeze(1).expand(
            h, w, -1
        )  # (h, w, d_model//2)
        pos_x = pos_x.unsqueeze(0).expand(
            h, w, -1
        )  # (h, w, d_model//2)

        pos_embed = torch.cat(
            [pos_y, pos_x], dim=-1
        )  # (h, w, d_model)
        pos_embed = pos_embed.reshape(
            h * w, self.d_model
        )  # (h*w, d_model)

        return pos_embed


class TransformerBlock(nn.Module):
    """Standard Transformer block with self-attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)

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
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm, attn_mask=attn_mask
        )
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x


class VisionARCModel(nn.Module):
    """Vision Transformer for ARC tasks."""

    def __init__(
        self,
        num_colors: int = 10,
        canvas_size: int = 64,
        patch_size: int = 2,
        d_model: int = 512,
        n_layers: int = 10,
        n_heads: int = 8,
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
        self.patch_proj = nn.Linear(
            patch_size * patch_size * d_model, d_model
        )

        # Task conditional tokens
        self.task_tokens = nn.Parameter(
            torch.randn(num_tasks, d_model) * 0.02
        )

        # Positional embeddings
        self.pos_embed = SeparablePositionalEmbedding2D(
            d_model, max_len=self.grid_size
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, mlp_dim, dropout)
                for _ in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)

        # Output projection: per-pixel classification
        self.output_proj = nn.Linear(
            d_model, num_colors + 2
        )  # Predict all token types

        self.dropout = nn.Dropout(dropout)

    def patchify(self, canvas: torch.Tensor) -> torch.Tensor:
        """
        Convert canvas to patches.

        Args:
            canvas: (B, H, W) with discrete color indices

        Returns:
            patches: (B, n_patches, patch_size^2 * d_model)
        """
        B, H, W = canvas.shape
        p = self.patch_size

        # Embed colors
        canvas_embed = self.color_embedding(
            canvas
        )  # (B, H, W, d_model)

        # Reshape to patches
        canvas_embed = canvas_embed.reshape(
            B, H // p, p, W // p, p, self.d_model
        )
        canvas_embed = canvas_embed.permute(
            0, 1, 3, 2, 4, 5
        )  # (B, H//p, W//p, p, p, d_model)
        patches = canvas_embed.reshape(
            B, (H // p) * (W // p), p * p * self.d_model
        )

        return patches

    def forward(
        self,
        canvas: torch.Tensor,
        task_id: int,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            canvas: (B, canvas_size, canvas_size) with discrete color indices
            task_id: Task identifier
            return_attention: Whether to return attention maps

        Returns:
            logits: (B, canvas_size, canvas_size, num_colors+2)
        """
        B = canvas.shape[0]

        # Patchify
        patches = self.patchify(
            canvas
        )  # (B, n_patches, patch_size^2 * d_model)

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
        """Create a new randomly initialized task embedding for test-time training."""
        return nn.Parameter(torch.randn(1, self.d_model) * 0.02)


class ARCDataAugmentation:
    """Data augmentation for ARC tasks."""

    @staticmethod
    def flip_horizontal(grid: torch.Tensor) -> torch.Tensor:
        return torch.flip(grid, dims=[-1])

    @staticmethod
    def flip_vertical(grid: torch.Tensor) -> torch.Tensor:
        return torch.flip(grid, dims=[-2])

    @staticmethod
    def rotate_90(grid: torch.Tensor) -> torch.Tensor:
        return torch.rot90(grid, k=1, dims=[-2, -1])

    @staticmethod
    def rotate_180(grid: torch.Tensor) -> torch.Tensor:
        return torch.rot90(grid, k=2, dims=[-2, -1])

    @staticmethod
    def rotate_270(grid: torch.Tensor) -> torch.Tensor:
        return torch.rot90(grid, k=3, dims=[-2, -1])

    @staticmethod
    def permute_colors(
        grid: torch.Tensor, permutation: List[int]
    ) -> torch.Tensor:
        """Apply color permutation."""
        result = grid.clone()
        for old_color, new_color in enumerate(permutation):
            result[grid == old_color] = new_color
        return result

    @staticmethod
    def get_augmentations():
        """Get all standard augmentations."""
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
    """Cross-entropy loss for ARC, ignoring background pixels."""

    def __init__(self, bg_token: int = 10):
        super().__init__()
        self.bg_token = bg_token

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-pixel cross-entropy loss.

        Args:
            logits: (B, H, W, C)
            targets: (B, H, W)

        Returns:
            loss: scalar
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


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = VisionARCModel(
        num_colors=10,
        canvas_size=64,
        patch_size=2,
        d_model=512,
        n_layers=10,
        n_heads=8,
        mlp_dim=512,
        dropout=0.1,
        num_tasks=400,
    )

    print(
        f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M"
    )

    # Example forward pass
    batch_size = 4
    dummy_canvas = torch.randint(0, 10, (batch_size, 64, 64))
    task_id = 0

    logits = model(dummy_canvas, task_id)
    print(f"Output shape: {logits.shape}")  # (B, 64, 64, 12)

    # Example loss computation
    dummy_target = torch.randint(0, 10, (batch_size, 64, 64))
    loss_fn = ARCLoss(bg_token=10)
    loss = loss_fn(logits, dummy_target)
    print(f"Loss: {loss.item():.4f}")
