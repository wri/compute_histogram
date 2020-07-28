from setuptools import setup

setup(
    name="compute_histogram",
    version="0.2.0",
    description="Tool to calculate histogram for input layer using optional mask layer",
    packages=["compute_histogram"],
    author="Thomas Maschler",
    license="MIT",
    install_requires=["click", "rasterio[s3]", "parallelpipe", "retrying"],
    entry_points="""
                [console_scripts]
                compute_histogram=compute_histogram.main:cli
                """,
)
