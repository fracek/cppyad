# Copyright 2020 Francesco Ceccon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import setuptools


project_root = Path(__file__).resolve().parent

about = {}
version_path = project_root / 'cppyad' / '__version__.py'
with version_path.open() as f:
    exec(f.read(), about)

with (project_root / 'README.rst').open() as f:
    readme = f.read()

with (project_root / 'CHANGELOG.rst').open() as f:
    changelog = f.read()


class get_pybind_include(object):
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


ext_modules = [
    Extension(
        'cppyad_core',
        [
            'src/module.cpp',
        ],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++',
    ),
]

setup(
    name='cppyad',
    description=about['__description__'],
    author=about['__author__'],
    author_email=about['__author_email__'],
    license=about['__license__'],
    version=about['__version__'],
    long_description=readme + '\n\n' + changelog,
    packages=find_packages(exclude=['tests']),
    ext_modules=ext_modules,
    install_requires=[
        'pyomo>=5.6.7',
        'numpy',
    ],
    setup_requires=['pytest-runner'],
    tests_require=[
        'pytest',
        'pytest-cov',
    ],
)
