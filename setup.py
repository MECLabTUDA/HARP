from setuptools import setup, find_packages

setup(
    name='HARPipe',
    version='1.1.2',
    packages=find_packages(),
    python_requires='>=3.11.0',
    install_requires=[
        'anomalib==0.7.0',
        'numpy',
        'pandas',
        'Pillow',
        'pytorch-lightning==1.9.5',
        'scikit-learn',
        'scipy',
        'scikit-image',
        'tifffile',
        'timm==0.6.5',
        'torch==2.0.1',
        'torchmetrics==0.10.3',
        'torchvision==0.15.2',
        'tqdm',
        'zarr'
    ],
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
