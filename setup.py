import os 
import setuptools 
from importlib import util as import_util

import setuptools.command
import setuptools.command.build_py
import setuptools.command.develop 

spec = import_util.spec_from_file_location('_metadata', 'muax/_metadata.py')
_metadata = import_util.module_from_spec(spec)
spec.loader.exec_module(_metadata)

muax_core_requirements = [
  'mctx',
  'dm-haiku', 
  'optax', 
  'gymnasium', 
  'lz4', 
  'tensorboardX'
]

tensorflow = [
  'tensorflow==2.8.0',
  'tensorflow_probability==0.15.0',
  'tensorflow_datasets==4.6.0',
  'dm-reverb==0.7.2',
  'dm-launchpad==0.5.2',
]

acme_core_requirements = [
    'dm-acme',
    'absl-py',
    'dm-env',
    'dm-tree',
    'numpy',
    'pillow',
    'typing-extensions',
]

acme_jax_requirements = [
  'jax==0.4.3',
  'jaxlib==0.4.3',
  'chex==0.1.6',
  'dm-haiku==0.0.10',
  'flax',
  'optax==0.1.7',
  'rlax==0.1.6',
] + tensorflow + acme_core_requirements

acme_tf_requirements = [
    'dm-sonnet',
    'trfl',
] + tensorflow + acme_core_requirements

testing_requirements = [
    'pytype==2023.12.8', 
    'pytest-xdist',
]

envs_requirements = [
    'atari-py',
    'bsuite',
    'dm-control',
    'gym==0.25.0',
    'gym[atari]',
    'pygame==2.1.0',
    'rlds',
]


def generate_requirements_file(path=None):
  """Generates requirements.txt file with the Acme's dependencies.

  Function from acme setup.py.
  It is used by Launchpad GCP runtime to generate Acme requirements to be
  installed inside the docker image. Acme itself is not installed from pypi,
  but instead sources are copied over to reflect any local changes made to
  the codebase.

  Args:
    path: path to the requirements.txt file to generate.
  """
  if not path:
    path = os.path.join(os.path.dirname(__file__), 'muax/requirements.txt')
  with open(path, 'w') as f:
    for package in set(muax_core_requirements 
                       + acme_core_requirements 
                       + acme_jax_requirements 
                       + acme_tf_requirements 
                       + envs_requirements):
      f.write(f'{package}\n')

with open('README.md', 'r') as f:
  long_description = f.read()

version = _metadata.__version__  


class BuildPy(setuptools.command.build_py.build_py):

  def run(self):
    generate_requirements_file()
    setuptools.command.build_py.build_py.run(self)

class Develop(setuptools.command.develop.develop):

  def run(self):
    generate_requirements_file()
    setuptools.command.develop.develop.run(self)

cmdclass = {
    'build_py': BuildPy,
    'develop': Develop,
}
  
setuptools.setup(
  name='muax',
  version=version,
  cmdclass=cmdclass,
  author = 'bwfbowen',
  description="A library that provides help for using MCTS RL with different frameworks.",
  keywords='reinforcement-learning mcts python muzero machine learning',
  long_description=long_description,
  long_description_content_type='text/markdown',
  packages=setuptools.find_packages(),
  package_data={"": ['requirements.txt']},
  include_package_data=True, 
  install_requires=muax_core_requirements,
  extras_require={
    'acme-jax': acme_jax_requirements,
    'acme-tf': acme_tf_requirements,
    'testing': testing_requirements,
    'envs': envs_requirements,
  },
  classifiers=[
    'Programming Language :: Python :: 3',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent'
  ],
  dependency_links=[
    'https://storage.googleapis.com/jax-releases/jax_releases.html',
  ],
  python_requires='>=3.9',
)
  
