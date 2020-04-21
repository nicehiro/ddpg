import setuptools

with open("README.org", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ddpg_pytorch-HIRO", # Replace with your own username
    version="0.0.1",
    author="Hiro",
    author_email="wfy11235813@gmail.com",
    description="ddpg implement using pytorch.",
    long_description=long_description,
    long_description_content_type="text/org",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
