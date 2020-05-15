import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="decision-tree-morfist",
    version="0.1.1",
    description="Multi-target Random Forest implementation that can mix both classification and regression tasks.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/systemallica/morfist",
    author="Andrés Reverón Molina",
    author_email="andres@reveronmolina.me",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    packages=["morfist"],
    install_requires=["numpy", "numba", "scipy"],
)
