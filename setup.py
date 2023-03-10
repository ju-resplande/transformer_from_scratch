from setuptools import setup, find_packages

setup(
    name="transformer",
    version="1.0.0",
    description="nn.Transformer implementation",
    url="https://github.com/ju-resplande/transformer_from_scratch",
    author="Juliana Resplande",
    classifiers=[
        "Development Status :: 6 - Mature",
        "Intended Audience :: Information Technology",
        "Environment :: GPU :: NVIDIA CUDA",
        "Topic :: Software Development :: Build Tools",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.6",
    ],
    keywords="transformers, dnn, deep neural networks",  # Optional
    package_dir={"": "src"},  # Optional
    packages=find_packages(where="src"),  # Required
    python_requires=">=3.6",
    install_requires=[
        "torch",
    ],
)
