from setuptools import setup, find_packages
from os import path

from text2phenotype.common.test_command import TestCommand


package_dir = path.abspath(path.dirname(__file__))
long_description = 'https://gettext2phenotype.atlassian.net/projects/BIOMED/versions/11243'


def parse_requirements(file_path):
    with open(file_path, 'r') as reqs:
        req_list = []
        for line in reqs.readlines():
            if line.startswith('#'):
                continue
            if line.startswith('-r '):
                sub_req = parse_requirements(f"{line.split('-r ')[-1].replace('/n', '').strip()}")
                req_list.extend(sub_req)
                continue
            req_list.append(line)
        return req_list


setup(
    name='biomed',
    version='1.2.0.2',
    description=long_description,
    long_description=long_description,
    url='',
    author='',
    author_email='',
    license='Other/Proprietary License',
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Healthcare Industry',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    keywords=['Text2phenotype BioMed', 'python'],
    packages=find_packages(exclude=['tests']),
    install_requires=parse_requirements('requirements.txt'),
    package_data={'': ['open_api.yaml']},
    include_package_data=True,
    entry_points={'console_scripts': ['biomed=biomed.__main__:main']},
    tests_require=['pytest'],
    cmdclass={'test': TestCommand}
)
