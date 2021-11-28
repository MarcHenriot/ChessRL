from setuptools import setup

setup(
    name='ChessRL',
    version='0.1',
    packages=['ChessRL', 'ChessRL.agent', 'ChessRL.environment', 'ChessRL.model', 'ChessRL.replayBuffer'],
    url='https://github.com/PeopleOfPlay/ChessRL.git',
    license='MIT',
    author='Marc Henriot, Adrien Turchini',
    description='Package for our final project of IFT_7201 at ULaval'
)