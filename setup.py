import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="waveml",
    version="0.2.0",
    author="leffff",
    author_email="levnovitskiy@gmail.com",
    description="Open source machine learning library with various models and tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leffff/waveml",
    license='Apache License 2.0',
    packages=setuptools.find_packages(),
    install_requires=["numpy", "pandas", "scikit-learn", "matplotlib", "torch", "torchvision"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)