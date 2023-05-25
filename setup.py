from setuptools import setup, find_packages

# Call setup function
setup(
    author="Vu Duc Tung",
    description="A package for downstream data analysis of proteomics data",
    name="proteomics_downsream_analysis",
    version="0.1.0",
    packages=find_packages(include=["proteomics_downsream_analysis", "proteomics_downsream_analysis.*"])
)
