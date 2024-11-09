from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path:str) -> List[str]:
	"""
		This function collects the packages from 'requirements.txt' and returns a list containing these packages
	"""

	if not file_path:
		raise FileNotFoundError(f'Please provide file path')
	
	requirements = []
	with open(file=file_path) as file_obj:
		requirements = file_obj.readlines()
		requirements = [req.replace('\n','') for req in requirements]

		if '-e .' in requirements:
			requirements.remove('-e .')
	return requirements

setup(
    name="car_price_prediction_system",  # Package name
    version="1.0.0",  # Version of the project
    author="Sushant Manglekar",
    author_email="manglekars145@gmail.com",
    description="A system to predict car prices based on various features",
    long_description=open("README.md").read(),  # Description can be the content of your README file
    long_description_content_type="text/markdown",
    url="https://github.com/SushantManglekar/car-price-prediction-system.git",  # URL to the project's GitHub or website
    packages=find_packages(),  # Automatically finds packages in the directory
    install_requires= get_requirements('requirements.txt'),
    entry_points={
        "console_scripts": [
            "car_price_predictor=car_price_prediction.main:run",  # Assumes you have a `run` function in main.py for running the project
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',  # Specify Python version requirement
)
