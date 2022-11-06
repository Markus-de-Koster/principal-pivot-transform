from setuptools import find_packages, setup


setup(
    name='PrincipalPivotTransform',
    packages=find_packages(include=['ppt']),
    version='0.1.0',
    description='(generalized) Principal Pivot Transform',
    author='Markus de Koster',
    license='MIT',
    test_suite="tests",
)
