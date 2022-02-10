from setuptools import setup, find_packages

# Load the README file.
with open(file="README.md", mode="r") as readme_handle:
    long_description = readme_handle.read()

classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: BSD License",
    "Operating System :: OS Independent"
]

# specify installation requirements
with open("requirements.txt") as requirements_file:
    install_requires = requirements_file.read().splitlines()

setup(
    name="pyfinch",
    version="0.0.1",
    description="A python package for analyzing neural & bioacoustics signals from songbirds",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JaerongA/pyfinch",
    author="Jaerong Ahn",
    author_email="jaerongahn@gmail.com",
    license="BSD 3-Clause License",
    packages=find_packages(include=['pyfinch']),
    include_package_data=True,
    install_requires=install_requires,
    python_requires='>=3.7',
    keywords=['python', 'songbird', 'zebra finch', 'neuroscience', 'bioacoustics'],
    classifiers=classifiers
)
