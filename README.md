# Tutorial 08 — Generative Adversarial Networks (GANs)
## Teaching Two Neural Networks to Outsmart Each Other
---

## Overview

This tutorial covers Generative Adversarial Networks from theory to implementation:

- The minimax game formulation and Nash equilibrium
- DCGAN Generator and Discriminator architecture (PyTorch)
- The alternating training loop with proper gradient blocking
- Mode collapse — diagnosis and solutions (WGAN, spectral norm, minibatch discrimination)
- Notable GAN variants: DCGAN, WGAN, Pix2Pix, CycleGAN, StyleGAN2
- FID score progress from 80 (DCGAN 2015) to 2.8 (StyleGAN2 2020)

---

## Files

| File | Description |
|------|-------------|
| `tutorial_08_gans.pdf` | Primary submission — tutorial PDF (<2000 words, 5 figures) |
| `tutorial_08_gans.docx` | Word source document |
| `tutorial_08_gans.ipynb` | Jupyter notebook — full runnable code, alt-text, references |
| `README.md` | This file |
| `LICENSE` | MIT Licence |
| `fig1_gan_architecture.png` | GAN flow diagram (Generator vs Discriminator) |
| `fig2_minimax_training.png` | Minimax value function + training dynamics |
| `fig3_dcgan_architecture.png` | DCGAN block diagram + latent space clusters |
| `fig4_mode_collapse.png` | Mode collapse scatter + solutions bar chart |
| `fig5_gan_variants.png` | Stability vs capability scatter + FID timeline |

---

## How to Run

### Requirements

```bash
pip install numpy matplotlib scipy jupyter
# For full DCGAN training on MNIST:
pip install torch torchvision
```

### Launch

```bash
jupyter notebook tutorial_08_gans.ipynb
```

Run all cells top to bottom (`Kernel → Restart & Run All`). All 5 figures regenerate and save as `.png` files. The full PyTorch DCGAN implementation is shown as reference code training MNIST requires a GPU (~30 min) but all figures run on CPU in under 1 minute.

---

## Key Architecture

```
Generator:  z(100) → Linear(100, 128×7×7) → ConvT(64,14×14) → ConvT(1,28×28,Tanh)
Discriminator: 28×28 → Conv(64,14×14,LReLU) → Conv(128,7×7,LReLU) → Linear → Sigmoid
```

Training: Adam(lr=2e-4, β=(0.5,0.999)) for both G and D.

---

## Accessibility

- **Colourblind-safe palette**  deep indigo (`#2D1B69`) and electric lime (`#A8E63D`) pass WCAG AA under deuteranopia, protanopia and tritanopia
- **Hatch patterns** on all bar charts (`//`, `xx`, `\\`, `..`) — information never by colour alone
- **Distinct marker shapes** on scatter plots (circle, square, triangle, diamond, down-triangle, plus)
- **Alt-text captions** printed below every figure cell in the notebook
- **Structured H1 → H2 heading hierarchy** for screen-reader navigation
- **High-contrast** dark (`#0D0A1A`) on light (`#FAFAF8`) — contrast ratio >14:1

---

## References

1. Goodfellow, I. et al. (2014) 'Generative adversarial nets', NeurIPS 27. https://arxiv.org/abs/1406.2661
2. Radford, A., Metz, L. and Chintala, S. (2015) 'DCGAN', arXiv:1511.06434. https://arxiv.org/abs/1511.06434
3. Arjovsky, M., Chintala, S. and Bottou, L. (2017) 'Wasserstein GAN', ICML 2017. https://arxiv.org/abs/1701.07875
4. Karras, T., Laine, S. and Aila, T. (2019) 'StyleGAN', CVPR 2019. https://arxiv.org/abs/1812.04948
5. Isola, P. et al. (2017) 'Pix2Pix', CVPR 2017. https://arxiv.org/abs/1611.07004
6. Zhu, J.Y. et al. (2017) 'CycleGAN', ICCV 2017. https://arxiv.org/abs/1703.10593

---

**Licence:** MIT — see `LICENSE`
