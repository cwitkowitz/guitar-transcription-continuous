from setuptools import setup, find_packages

setup(
    name='guitar-transcription-continuous',
    url='https://github.com/cwitkowitz/guitar-transcription-continuous',
    author='Frank Cwitkowitz',
    author_email='fcwitkow@ur.rochester.edu',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=['amt_tools', 'muda'],
    version='0.0.2',
    license='MIT',
    description='Code for continuous guitar transcription',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown'
)
