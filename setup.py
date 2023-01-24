import setuptools 

with open('README.md', 'r') as f:
  long_description = f.read()
  
setuptools.setup(
  name='muax',
  version='0.0.2.2',
  authors = [{ 'name': "Bowen Fang", 'email': "bf2504@columbia.edu" },
             {'name': 'Ian Chie', 'email': 'cc4893@columbia.edu'}],
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
  install_requires=['mctx', 'dm-haiku', 'optax']
)
  
