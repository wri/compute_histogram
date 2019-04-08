from setuptools import setup

setup(
    name="compute_histogram",
    version="0.1.0",
    description="Tool to calculate histogram for input layer using optional mask layer",
    packages=["compute_histogram"],
    author="Thomas Maschler",
    license="MIT",
    install_requires=["rasterio"],
    scripts=["compute_histogram/main.py"],
)
