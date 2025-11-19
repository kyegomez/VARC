import torch
from varc.main import VisionARCModel, ARCLoss

# Example usage
if __name__ == "__main__":
    # Initialize model with grouped query attention
    model = VisionARCModel(
        num_colors=10,
        canvas_size=64,
        patch_size=2,
        d_model=512,
        n_layers=10,
        n_query_heads=8,  # Number of query heads
        n_kv_heads=2,  # Number of key-value heads (shared across query groups)
        mlp_dim=512,
        dropout=0.1,
        num_tasks=400,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

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
