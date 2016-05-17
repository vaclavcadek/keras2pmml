from distutils.core import setup

setup(name='keras2pmml',
      version='0.0.1',
      description='Simple exporter of Keras models into PMML',
      author='Francois Chollet',
      author_email='vaclav.cadek@gmail.com',
      url='https://github.com/vaclavcadek/keras2pmml',
      license='MIT',
      install_requires=['theano', 'keras', 'scikit-learn'],
      extras_require={},
      packages=['keras2pmml']
      )
