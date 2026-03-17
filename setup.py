from setuptools import setup, find_packages
def readme():
    with open('README.md') as f:
        return f.read()

dependencies = [
    'pandas>=1.1.1',
    'numpy>=1.3.0',
    'scikit-learn>=0.24.0',
    'tqdm>=4.40.0',
    'statsmodels>=0.10.2',
    'lightgbm>=3.0.0',
    'xgboost>=1.0.0',
    'matplotlib>=3.3.0',
    'FixedEffectModel>=0.0.5'
]

setup(
    name="causalmatch",
    version='0.0.6',
    author="Xiaoyu Zhou",
    author_email="xiaoyuzhou@bytedance.com",
    url='https://github.com/bytedance/CausalMatch',
    description="Propensity score matching and coarsened exact matching",
    packages=find_packages(),
    install_requires=dependencies,
    python_requires='>=3.6'
)
