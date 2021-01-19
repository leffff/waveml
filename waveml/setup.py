import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = f.readlines()

setuptools.setup(
    name="waveml",
    version="0.1",
    author="leffff",
    author_email="levnovitskiy@gmail.com",
    description="Open source machine learning library for performance of a weighted average over stacked predictions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leffff/waveml",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache-2.0 License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)