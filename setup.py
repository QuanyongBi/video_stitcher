from setuptools import setup, find_packages

setup(
    name="video_stitcher",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "pillow>=8.0.0",
    ],
    python_requires=">=3.8",
    author="Quanyong Bi, Zhehao Xu",
    author_email="quanyong@usc.edu",
    description="A tool for creating ultra-high-resolution images from videos, a final project for USC CSCI576 2025 Spring.",
    keywords="video, stitching, panorama, image processing",
    url="https://github.com/yourusername/video-stitcher",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)