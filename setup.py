from setuptools import setup, find_packages

# call setup function
setup(
    author='Dinara Issagaliyeva',
    description='Landmark classifier using PyTorch and FastAI to distinguish 50 places.',
    long_description=open('README.md').read() + '\n\n' + open('CHANGELOG.txt').read(),
    name='landmark_classifier',
    version='0.1.0',
    license='MIT',
    install_requires=[
        'pandas', 'numpy', 'matplotlib', 'seaborn',
        'fastai', 'pytorch', 'torchvision', 'torchsummary',
        'split-folders', 'pillow~=9.0.1'
    ]
)
