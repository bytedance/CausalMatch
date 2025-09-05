from setuptools import setup, find_packages
def readme():
    with open('README.md') as f:
        return f.read()

dependencies = [
    'pandas>=1.3.4, <=1.3.5',
    'numpy>=1.3.0, <=1.26.4',
    'scikit-learn>=1.4.0',
    'tqdm>=4.66.2',
    'statsmodels>=0.14.1',
    'lightgbm>=4.3.0',
    'xgboost>=2.0.3',
    'matplotlib>=3.8.3',
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
