from setuptools import find_packages, setup

setup(
    name='satya',
    version='1.0.0',
    description='Satya: A Chrome extension for detecting YouTube comment sentiments.',
    author='Vivek Kumar',
    author_email='mrvivekkumar7171@gmail.com',
    url='https://github.com/mrvivekkumar7171/satya',
    license='Proprietary',
    packages=find_packages(where="backend"),
    package_dir={"": "backend"},
    install_requires=[
        "dvclive"
        "PyYAML"
        "wordcloud"
        "dvc"
        "seaborn"
        "nltk"
        "awscli"
        "dvc-s3"
        "mlflow"
        "cloudpickle"
        "lightgbm"
        "matplotlib"
        "numpy"
        "pandas"
        "psutil"
        "pyarrow"
        "scikit-learn"
        "scipy"
        "flask_cors"
        "flask"
        "joblib"
        "docutils"
        "pytest"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.11.13",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11.13",
)