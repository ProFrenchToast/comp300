import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="comp300",
    version="1.0.0",
    author="Patrick Quinn",
    author_email="Patrick.Quinn-2@student.machester.ac.uk",
    description="A package to implement inverse reinforcement learning. Created as part of COMP300 project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.cs.man.ac.uk/f46471pq/comp300",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)