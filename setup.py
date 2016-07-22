# coding=utf-8
import os
from setuptools import setup

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='keras2pmml',
    version='0.0.1',
    packages=['keras2pmml'],
    include_package_data=True,
    license='MIT',
    description='Simple exporter of Keras models into PMML',
    url='https://github.com/vaclavcadek/keras2pmml',
    author='Václav Čadek',
    author_email='vaclavcadek@gmail.com',
    install_requires=['theano', 'keras', 'scikit-learn'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Development Status :: 3 - Alpha',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
