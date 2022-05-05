from setuptools import setup, find_packages

# call setup function
setup(
    author='Dinara Issagaliyeva',
    description='Landmark classifier using PyTorch and FastAI to distinguish 50 places.',
    long_description=open('README.md').read() + '\n\n' + open('CHANGELOG.txt').read(),
    name='landmark_classifier',
    version='0.1.0',
    # install_requires=['pandas', 'scipy', 'matplotlib'],
    install_requires=[
        'pandas>=1.0',
        # 'matplotlib>=2.2.1,<3' # bigger than 2.2.1 but < 3
    ]
)
