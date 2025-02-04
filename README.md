# SLiM: One-shot Quantized Sparse Plus Low-rank Approximation of LLMs

This repository contains the implementation of SLiM (Sparse Low-rank Approximation with Quantization), a novel 
compression technique for large language models (LLMs). SLiM combines a one-shot quantization and sparse low-rank 
approximation to reduce memory usage and improve inference speed without requiring retraining. The approach features 
SLIM-Quant, a symmetric quantization method, and a saliency-based low-rank approximation that leverages sparsity 
patterns like 2:4 for optimized performance on accelerated hardware. With this, SLiM offers state-of-the-art accuracy 
while maintaining efficiency in memory-constrained environments.

**SLiM: One-shot Quantized Sparse Plus Low-rank Approximation of LLMs**

*Mohammad Mozaffari, Amir Yazdanbakhsh, and Maryam Mehri Dehnavi*

Paper: [https://arxiv.org/abs/2410.09615](https://arxiv.org/abs/2410.09615)

## Code Coming Soon!
We are excited to share our code with the community and are working on preparing it for release. Please stay tuned for updates, and thank you for your patience!


## Citation
If you use SLiM in your research, please cite our paper:
```angular2html
@article{slim:2024,
    title        = {{SLiM: One-shot Quantized Sparse Plus Low-rank Approximation of LLMs}},
    author       = {Mozaffari, Mohammad and Mehri Dahnavi, Maryam},
    year         = 2024,
    journal      = {arXiv preprint}
}
```
