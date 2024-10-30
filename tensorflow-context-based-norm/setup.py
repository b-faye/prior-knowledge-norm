from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="tensorflow_context_based_norm",
    version="1.0.0",
    author="Bilal FAYE",
    author_email="faye@lipn.univ-paris13.fr",
    url="https://github.com/b-faye/prior-based-norm",
    description="Prior-Based Normalization provides versatile normalization layers for deep neural networks, including ContextNorm, ContextExtendedNorm and AdaptiveContextNorm versions. Enhance model generalization and robustness by efficiently integrating prior knowledge.",    
    packages=find_packages(),
    readme="README.md",
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=["tensorflow >= 2.17.0"],
    python_requires=">=3.10",
    classifiers=[
    "Programming Language :: Python :: 3.10",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules"
    ]


)
