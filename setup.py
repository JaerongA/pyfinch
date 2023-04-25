from setuptools import find_packages, setup

with open(file="README.md", mode="r") as readme_handle:
    long_description = readme_handle.read()
    
with open("version.py") as version_file:
    exec(version_file.read())
    
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: BSD License",
    "Operating System :: OS Independent",
]

setup(
    name="pyfinch",
    version=__version__,
    description="A python package for analyzing neural & bioacoustics signals from songbirds",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JaerongA/pyfinch",
    author="Jaerong Ahn",
    author_email="jaerongahn@gmail.com",
    license="BSD 3-Clause License",
    packages=['pyfinch'],
    include_package_data=True,
    python_requires=">=3.7",
    keywords=["python", "songbird", "zebra finch", "neuroscience", "bioacoustics"],
    classifiers=classifiers,
    # Load package dependencies from environment.yml file.
    # Note that this requires the `conda-pack` package to be installed.
    # Alternatively, you can use `conda env export --file environment.yml`
    # to generate the environment.yml file and manually update the `setup.py` file.
    setup_requires=['conda-pack'],
    install_requires=['conda-pack'],
    extras_require={
        'environment': [
            'conda-build',
            'conda-verify'
        ]
    },
    zip_safe=False,
    options={
        'conda': {
            'yaml_file': 'environment.yml'
        }
    },
)