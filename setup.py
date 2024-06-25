from setuptools import setup, find_packages

setup(
    name='HARPipe',
    version='1.0.1',
    packages=find_packages(),
    install_requires=[],  # List your project's dependencies here
    description='Histological Artifact Restoration Pipeline',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='MECLabTUDa',
    author_email='moritz.fuchs@gris.tu-darmstadt.de',
    url='https://github.com/MECLabTUDA/HARP',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: CC BY 4.0',
        'Operating System :: OS Independent',
    ],
)
