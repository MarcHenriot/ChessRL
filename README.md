
# Reinforcement Learning Chess
#### Marc Henriot, Adrien Turchini

## Features

### Capture Chess 
- Goal: Capture as many pieces.
- Motivation: Easy to implement, gives a good basis for the agent.
- Concepts: Q-learning, value function approximation, experience replay, fixed-q-targets, policy gradients.

## Installation
```bash
pip install git+https://github.com/PeopleOfPlay/ChessRL.git
```

## ATTENTION ! 
Pour Windows : l'agent est disponible en ligne sur le site officiel et est donc compatbile sur toutes les machines, le programme devrait compiler seul.

Pour MacOS : l'agent n'est pas disponible, nous avons donc été cherché le programme Stockfish installé dans la librairie stockfish de Brew. N'ayant pas de moyen de tester le fonctionnement sur un autre Mac il se peut que le programme ne compile pas seul. Dans ce cas il s'agirait alors d'installer Stockfish via brew puis d'aller chercher l'executable dans le la librairie installée par Brew selon l'emplacement attitré sur votre ordinateur.
```bash
brew install stockfish
```

Pour linux : nous avons mis à disposition un agent disponible sur le site officiel de Stockfish mais n'avons pu testé la solution sur cet OS.