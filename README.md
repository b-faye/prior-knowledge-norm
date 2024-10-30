# Context-Based Normalization

We propose 03 normalization methods based on prior knowledge called contexts:

- **Context Normalization (CN)**


- **Context Normalization Extended (CN-X)**


- **Adaptive Context Normalization (ACN)**


## References


- **All versions:** *Enhancing Neural Network Representations with Prior Knowledge-Based Normalization*, FAYE et al., [ArXiv Link](https://arxiv.org/abs/2403.16798)


## Usage

### Tensorflow
- For manual installation, navigate to the directory named "tensorflow-context-based-norm".
    ```bash
    git clone git@github.com:b-faye/prior-knowledge-norm.git
    cd tensorflow-context-based-norm
    pip install dist/tensorflow_context_based_norm-1.0.tar.gz
    ```

- For online installation, please follow the provided instructions:
    ```bash
    pip install tensorflow-context-based-norm
    ```
    ```python
    from tensorflow_context_based_norm import ContextNorm, ContextExtendedNorm, AdaptiveContextNorm
    context_norm = ContextNorm(num_contexts=2)
    context_extended_norm = ContextExtendedNorm(num_contexts=10)
    adaptive_context_norm = AdaptiveContextNorm(num_contexts=3)
    ```

### Keras
- For manual installation, navigate to the directory named "keras-context-based-norm".
    ```bash
    git clone git@github.com:b-faye/prior-knowledge-norm.git
    cd keras-context-based-norm
    pip install dist/keras_context_based_norm-1.0.tar.gz
    ```
- For online installation, please follow the provided instructions:
    ```bash
    pip install keras-context-based-norm
    ```
    ```python
    from keras_context_based_norm import ContextNorm, ContextExtendedNorm, AdaptiveContextNorm
    context_norm = ContextNorm(num_contexts=2)
    context_extended_norm = ContextExtendedNorm(num_contexts=10)
    adaptive_context_norm = AdaptiveContextNorm(num_contexts=3)
    ```

### PyTorch
- For manual installation, navigate to the directory named "pytorch-context-based-norm".
    ```bash
    git clone git@github.com:b-faye/prior-knowledge-norm.git
    cd pytorch-context-based-norm
    pip install dist/pytorch_context_based_norm-1.0.tar.gz
    ```

- For online installation, please follow the provided instructions:
    ```bash
    pip install pytorch-context-based-norm
    ```
    ```python
    from pytroch_context_based_norm import ContextNorm, ContextExtendedNorm, AdaptiveContextNorm
    context_norm = ContextNorm(num_contexts=2)
    context_extended_norm = ContextExtendedNorm(num_contexts=10)
    adaptive_context_norm = AdaptiveContextNorm(num_contexts=3)
    ```


## Citing Neural Network Representations with Prior Knowledge-Based Normalization

If you find our contribution useful in your research, please consider citing:

```bash
@article{faye2024enhancing,
  title={Enhancing Neural Network Representations with Prior Knowledge-Based Normalization},
  author={Faye, Bilal and Azzag, Hanane and Lebbah, Mustapha and Bouchaffra, Djamel},
  journal={arXiv preprint arXiv:2403.16798},
  year={2024}
}
```
