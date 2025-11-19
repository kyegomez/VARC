# VARC: Vision ARC

![Architecture](/architecture.png)

This is a clean, open-source, single-file implementation of the model architecture from the ["ARC Is a Vision Problem"](https://arxiv.org/abs/2511.14761) paper. It fully features the paper's Canvas mechanism and a fast transformer with grouped-query attention + RMSNorm.


[![Discord](https://img.shields.io/badge/Discord-Join%20Community-5865F2?logo=discord&logoColor=white)](https://discord.gg/EamjgSaEQf)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-2511.14761-b31b1b.svg)](https://arxiv.org/abs/2511.14761)


## Key Contributions

| Key Contribution                  | Description                                                                                                                   |
|-----------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| **Vision-centric formulation**    | Reformulates ARC as an image-to-image translation problem rather than a language-oriented task                                 |
| **Canvas representation**         | Introduces a canvas-based representation that enables processing ARC tasks as natural images                                  |
| **Standard vision architecture**  | Demonstrates that a vanilla Vision Transformer can effectively solve ARC tasks when trained from scratch                      |
| **Test-time training**            | Employs test-time training for generalization to unseen tasks                                                                 |
| **Strong performance**            | Achieves 60.4% accuracy on ARC-1, outperforming existing from-scratch methods and competing with leading LLMs                |

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
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
        n_kv_heads=2,      # Number of key-value heads (shared across query groups)
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

```

See `example.py` for a complete usage example.

## Documentation

For comprehensive API reference documentation, see [VisionARCModel Reference](VisionARCModel_reference.md).

## Architecture

The VARC framework consists of several key components:

- **Canvas**: Handles placement of ARC grids on a fixed-size canvas with scaling and translation
- **Vision Transformer**: Standard ViT architecture adapted for image-to-image translation
- **Separable Positional Embeddings**: 2D separable positional embeddings for spatial awareness
- **Task Conditioning**: Task-specific tokens for multi-task learning
- **Data Augmentation**: Geometric and color-based augmentations for improved generalization

## Results

- **ARC-1 Accuracy**: 60.4%
- **Training**: From scratch on ARC data only
- **Generalization**: Test-time training for unseen tasks

## Citation

If you use this code in your research, please cite:

```bibtex
@article{hu2025arc,
  title={ARC Is a Vision Problem!},
  author={Hu, Keya and Cy, Ali and Qiu, Linlu and Ding, Xiaoman Delores and Wang, Runqian and Zhu, Yeyin Eva and Andreas, Jacob and He, Kaiming},
  journal={arXiv preprint arXiv:2511.14761},
  year={2025}
}
```

## License

See [LICENSE](LICENSE) file for details.

## Acknowledgments

This is a community implementation of the VARC framework. For the official implementation and project webpage, please refer to the [original repository](https://github.com/lillian039/VARC).
