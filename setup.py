from setuptools import setup, find_packages

# Read README.md safely
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except UnicodeDecodeError:
        # Fallback if UTF-8 fails
        try:
            with open("README.md", "r", encoding="latin-1") as fh:
                return fh.read()
        except:
            return "A Python wrapper for the Statistics Sweden PxWebAPI 2.0"
    except:
        return "A Python wrapper for the Statistics Sweden PxWebAPI 2.0"

setup(
    name="pxstatspy",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "requests>=2.25.0",
        "pandas>=1.0.0",
    ],
    python_requires=">=3.7",
    author="Emanuel Raptis",
    description="A Python wrapper for the Statistics Sweden PxWebAPI 2.0",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/xemarap/pxstatspy",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="statistics, sweden, api, data, px",
)