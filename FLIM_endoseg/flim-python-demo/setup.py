from setuptools import setup, find_packages
import subprocess


install_deps = [
        "torch",
        "torchvision",
        "pillow",
        "numpy",
        "matplotlib",
        "setuptools",
        "scikit-image",
        "scikit-learn",
        "faiss-cpu"
    ]

#gpu_available=False
# try:
#     subprocess.check_output('nvidia-smi')
#     install_deps.append('faiss-gpu')
# except Exception: # this command not being found can raise quite a few different errors depending on the configuration
#     install_deps.append('faiss-cpu')

with open("README.md", "r") as fh:
    long_description = fh.read()

__version__ = None
exec(open("pyflim/_version.py").read())

setup(
    name='pyflim',
    version='0.1',
    packages=find_packages(),
    install_requires=install_deps,
    author='Leonardo de Melo João',
    author_email='leomelo168@gmail.com',
    description='This is a python implementation of the Feature Learning by Image Markers.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/LIDS-UNICAMP/flim-python',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Education :: Developers",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: \
            Artificial Intelligence :: Image Recognition",
        "Topic :: Software Development :: Libraries",
    ],
    python_requires='>=3.10',
)
