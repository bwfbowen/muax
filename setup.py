import setuptools 

with open('README.md', 'r') as f:
  long_description = f.read()
  
setuptools.setup(
  name='muax',
  version='0.0.1',
  author='Bowen Fang',
  description="A library written in Jax that provides help for using DeepMind's mctx on gym-style environments.",
  long_description=long_description,
  packages=setuptools.find_packages(),
  classifiers=[
    'Programming Language :: Python :: 3',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent'
  ],
  python_requires='>=3.6',
  py_modules=['muax'],
  install_requires=['mctx', 'coax', 'dm-haiku', 'optax']
)
  
