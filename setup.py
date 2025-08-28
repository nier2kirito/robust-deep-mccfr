#!/usr/bin/env python3
"""
Setup script for Deep Learning Monte Carlo Counterfactual Regret Minimization (DL-MCCFR).
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Extract core requirements (remove version constraints and optional dependencies)
core_requirements = []
for req in requirements:
    if any(keyword in req.lower() for keyword in ['pytest', 'black', 'flake8', 'mypy', 'sphinx']):
        continue  # Skip development/documentation dependencies
    # Clean up version constraints for core requirements
    if '>=' in req:
        req = req.split('>=')[0]
    elif '==' in req:
        req = req.split('==')[0]
    core_requirements.append(req)

setup(
    name="dl-mccfr",
    version="1.0.0",
    author="Zakaria El-Jaafari",
    author_email="",
    description="Deep Learning Monte Carlo Counterfactual Regret Minimization for imperfect information games",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/DL_MCCFR",  # Update with actual URL
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=core_requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0", 
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
            "scipy>=1.7.0",
            "tqdm>=4.62.0",
            "tensorboard>=2.8.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "dl-mccfr-train=dl_mccfr.cli:train_cli",
            "dl-mccfr-evaluate=dl_mccfr.cli:evaluate_cli",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="machine-learning deep-learning game-theory poker mccfr counterfactual-regret-minimization",
    project_urls={
        "Bug Reports": "https://github.com/your-username/DL_MCCFR/issues",
        "Source": "https://github.com/your-username/DL_MCCFR",
        "Documentation": "https://dl-mccfr.readthedocs.io/",
    },
)
