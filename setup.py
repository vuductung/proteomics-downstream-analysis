from setuptools import setup, find_packages

# Call setup function
setup(
    name="proteomics_downstream_analysis",
    author="Vu Duc Tung",
    author_email="tungvuduc@outlook.de",
    description="A package for downstream data analysis of proteomics data",
    name="proteomics_downsream_analysis",
    keywords = ['proteomics', 'downstream analysis', 'data analysis', 'data visualization', 'mass spectrometry'],
    version="0.1.0",
    url="https://github.com/vuductung/proteomics-downstream-anlaysis",
    packages=find_packages(include=["proteomics_downstream_analysis", "proteomics_downstream_analysis.*"]),
    python_requires='3.11.3',
    install_requires=[
                    'adjustText',
                    'goatools',
                    'gseapy',
                    'matplotlib',
                    'matplotlib-venn',
                    'numba',
                    'numpy',
                    'plotly',
                    'pyarrow',
                    'pycirclize',
                    'pyteomics',
                    'scikit-learn',
                    'scipy',
                    'seaborn',
                    'statsmodels',
                    'streamlit',
                    'umap',
                    'UpSetPlot',
                    'venn'
                    ]
)