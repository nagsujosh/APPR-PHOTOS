from setuptools import setup, find_packages

setup(
    name="appr-photos",
    version="0.1.0",
    description="Adversary-Adaptive Representation Learning for Privacy-Preserving Image Tasks",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "Pillow",
        "tensorboard",
        "umap-learn",
        "matplotlib",
        "seaborn",
        "pyyaml",
        "tqdm",
        "gdown",
    ],
)
