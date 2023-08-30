from setuptools import find_packages
from setuptools import setup

# keep this in sync with requirements.txt

REQUIRED_PACKAGES = ['transformers',
                     'datasets',
                     'huggingface_hub',
                     'peft',
                     'accelerate',
                     'bitsandbytes',
                     'torch',
                     'scipy',
                     'google-cloud-storage']

setup(
    name='trainer',
    version='1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='LLM generated fanfiction',
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Michael Kuehl",
    author_email="hello@txs.vc",
    url="https://github.com/mickume/llm-stories",
    license="MIT",
    python_requires=">=3.9",
)
