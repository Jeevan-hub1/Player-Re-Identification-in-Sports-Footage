from setuptools import setup, find_packages

setup(
    name="player-reid",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "ultralytics>=8.0.0",
        "opencv-python>=4.5.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
    ],
    python_requires=">=3.8",
    author="Your Name",
    description="Player Re-identification in Sports Footage",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)