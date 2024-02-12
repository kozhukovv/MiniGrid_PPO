from pathlib import Path
from setuptools import setup, find_packages

description = ["Maingrid PPO"]

root = Path(__file__).parent
with open(str(root / "README.md"), "r", encoding="utf-8") as f:
    readme = f.read()
with open(str(root / "requirements.txt"), "r") as f:
    dependencies = f.read().split("\n")

setup(
    name="maingrid_ppo",
    version="0.0.1a",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=dependencies,
    author="@kozhukovv",
    description=description,
    long_description=readme,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
