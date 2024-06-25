from setuptools import setup, find_packages

setup(
    name='HARPipe',
    version='1.0',
    packages=find_packages(),
    python_requires='>=3.11.4',
    install_requires=[
        'anomalib',
        'numpy',
        'pandas',
        'Pillow',
        'pytorch-lightning',
        'scikit-learn',
        'scipy',
        'scikit-image',
        'tifffile',
        'timm',
        'torch',
        'torchmetrics',
        'torchvision',
        'tqdm',
        'zarr',
        'segment-anything @ git+https://github.com/facebookresearch/segment-anything.git'
    ]],  # List your project's dependencies here
    description='Histological Artifact Restoration Pipeline',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='MECLabTUDa',
    author_email='moritz.fuchs@gris.tu-darmstadt.de',
    url='https://github.com/MECLabTUDA/HARP',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved',
        'Operating System :: OS Independent',
    ],
)
