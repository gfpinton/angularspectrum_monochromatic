from setuptools import setup, find_packages

setup(
    name="angularspectrum",
    version="0.1.0",
    description="Angular Spectrum method for monochromatic wave propagation",
    author="",
    author_email="",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "gpu": ["torch>=1.9.0"],
    },
    python_requires=">=3.6",
) 