from setuptools import setup, find_packages


def parse_requirements(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="AC-VC-paper",
    version="v0.1",
    url="https://github.com/Iacaruso-lab/AC-VC-paper",
    license="MIT",
    author="Znamenskiy lab, Iacaruso lab",
    author_email="alexander.egea-weiss@crick.ac.uk, benita.turner-bridger@crick.ac.uk",
    description="Code for reproducing figures and analysis in Egea-Weiss*, Turner-Bridger* et al., (2025)",
    packages=find_packages(),
    include_package_data=True,
    install_requires=parse_requirements("requirements.txt"),
)
