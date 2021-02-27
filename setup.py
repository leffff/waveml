import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="waveml",
    version="0.1.10",
    author="leffff",
    author_email="levnovitskiy@gmail.com",
    description="Open source machine learning library for performance of a weighted average over stacked predictions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leffff/waveml",
    license='Apache License 2.0',
    packages=setuptools.find_packages(),
    install_requires=["numpy", "pandas", "scikit-learn", "matplotlib"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)