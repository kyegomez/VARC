# VARC: Vision ARC

![Architecture](/architecture.png)

**An open-source community implementation of "ARC Is a Vision Problem!"**

[![arXiv](https://img.shields.io/badge/arXiv-2511.14761-b31b1b.svg)](https://arxiv.org/abs/2511.14761)

## Paper

**ARC Is a Vision Problem!**  
Keya Hu, Ali Cy, Linlu Qiu, Xiaoman Delores Ding, Runqian Wang, Yeyin Eva Zhu, Jacob Andreas, Kaiming He

**arXiv:** [2511.14761](https://arxiv.org/abs/2511.14761) [cs.CV]  
**DOI:** [10.48550/arXiv.2511.14761](https://doi.org/10.48550/arXiv.2511.14761)

## Abstract

The Abstraction and Reasoning Corpus (ARC) is designed to promote research on abstract reasoning, a fundamental aspect of human intelligence. Common approaches to ARC treat it as a language-oriented problem, addressed by large language models (LLMs) or recurrent reasoning models. However, although the puzzle-like tasks in ARC are inherently visual, existing research has rarely approached the problem from a vision-centric perspective. 

In this work, we formulate ARC within a vision paradigm, framing it as an image-to-image translation problem. To incorporate visual priors, we represent the inputs on a "canvas" that can be processed like natural images. It is then natural for us to apply standard vision architectures, such as a vanilla Vision Transformer (ViT), to perform image-to-image mapping. Our model is trained from scratch solely on ARC data and generalizes to unseen tasks through test-time training. Our framework, termed Vision ARC (VARC), achieves **60.4% accuracy** on the ARC-1 benchmark, substantially outperforming existing methods that are also trained from scratch. Our results are competitive with those of leading LLMs and close the gap to average human performance.

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
