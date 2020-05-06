import setuptools

with open("README.md", 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name="dftools",
    version="0.2.5",
    author="Shane Breeze",
    author_email="sdb15@ic.ac.uk",
    scripts=[],
    description="dataframe tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy>=1.18.1",
        "pandas>=1.0.1",
        "matplotlib>=3.1.2",
        "scipy>=1.4.1",
        "numdifftools>=0.9.39",
        "iminuit>=1.3.8",
    ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
        "Development Status :: 3 - Alpha",
    ],
)
