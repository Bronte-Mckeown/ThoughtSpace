from setuptools import setup, find_packages
with open('requirements.txt') as f:
    required = f.read().splitlines()
setup(
   name='ThoughtSpace',
   version='0.0.1',
   description='Package of functions to run and visualize PCA on ESQ data, and project between datasets.',
   author='Bronte Mckeown & Will Strawson',
   author_email='bronte.mckeown@gmail.com',
   packages=find_packages(include=['ThoughtSpace']),
   install_requires=required,
   package_data={'ThoughtSpace': [
    'fonts/*.ttf'
    ]
    }
 
)
