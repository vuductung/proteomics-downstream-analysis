from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()
    
# Call setup function
setup(
    name="proteomics_downstream_analysis",
    author="Vu Duc Tung",
    author_email="tungvuduc@outlook.de",
    description="A package for downstream data analysis of proteomics data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords = ["proteomics", "downstream analysis", "data analysis", "data visualization", "mass spectrometry"],
    version="0.1.1",
    url="https://github.com/vuductung/proteomics-downstream-anlaysis",
    packages=find_packages(include=["repos","proteomics_downstream_analysis", "proteomics_downstream_analysis.*"]),
    python_requires=">=3.6.1",
    install_requires=[
                    "adjustText",
                    "goatools",
                    "gseapy",
                    "matplotlib",
                    "matplotlib-venn",
                    "numba",
                    "numpy",
                    "plotly",
                    "pyarrow",
                    "pycirclize",
                    "pyteomics",
                    "scikit-learn",
                    "scipy",
                    "seaborn",
                    "statsmodels",
                    "streamlit",
                    "umap",
                    "UpSetPlot",
                    "venn"
                    ]
)