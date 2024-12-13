from setuptools import setup, find_packages

setup(
    name="ml-model-benchmark-suite",
    version="0.1.0",
    description="Structured framework for training, evaluating, and comparing ML models",
    author="DigitalNomad",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.3.0",
        "xgboost>=1.7.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "shap>=0.42.0",
        "openml>=0.13.0",
        "jinja2>=3.1.0",
    ],
    entry_points={
        "console_scripts": [
            "ml-benchmark=main:main",
        ],
    },
)
