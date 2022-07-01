import setuptools
import sys
import os

try:
    import builtins
except ImportError:
    import __builtin__ as builtins

PATH_ROOT = PATH_ROOT = os.path.dirname(__file__)
builtins.__clhive_SETUP__ = True


def load_description(path_dir=PATH_ROOT, filename="README.md"):
    """Load long description from readme in the path_dir/ directory
    """
    with open(os.path.join(path_dir, filename)) as f:
        long_description = f.read()
    return long_description


def load_requirements(path_dir=PATH_ROOT, filename="requirements.txt"):
    """Load requirements from text file in the path_dir/requirement.txt
    """
    with open(os.path.join(path_dir, filename), "r") as f:
        packages = f.read().splitlines()
    return packages


if __name__ == "__main__":

    name = "clhive"
    version = "0.1.0"
    description = "CLHive is a PyTorch framework for Continual Learning research."
    author = "Nader Asadi"
    author_email = "asadi.nader97@gmail.com"

    python_requires = ">=3.6"
    install_requires = load_requirements()

    project_urls = {
        "Github": "https://github.com/naderasadi/CLHive",
    }

    classifiers = [
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
    ]

    setuptools.setup(
        name=name,
        version=version,
        author=author,
        author_email=author_email,
        description=description,
        license="MIT",
        long_description=load_description(),
        long_description_content_type="text/markdown",
        install_requires=install_requires,
        python_requires=python_requires,
        packages=setuptools.find_packages(),
        classifiers=classifiers,
        include_package_data=True,
        project_urls=project_urls,
    )
