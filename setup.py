"""
Setup configuration for Hegarty package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="hegarty",
    version="0.1.0",
    author="Hegarty Research Team",
    description="A perspective-taking agent for enhanced spatial reasoning using GPT-4o and Sora-2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Hegarty-0.1",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "openai>=1.6.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "httpx>=0.24.0",
        "opencv-python>=4.8.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ]
    },
    zip_safe=False,
)
