
# Reinforcement Learning Chess
#### Marc Henriot, Adrien Turchini

## ATTENTION Stockfish ! 
Pour Windows : l'agent Stockfish est disponible en ligne sur le site officiel et est donc compatbile sur toutes les machines, le programme devrait compiler seul.

Pour MacOS : l'agent Stockfish n'est pas disponible, nous avons donc été cherché le programme Stockfish installé dans la librairie stockfish de Brew. N'ayant pas de moyen de tester le fonctionnement sur un autre Mac il se peut que le programme ne compile pas seul. Dans ce cas il s'agirait alors d'installer Stockfish via brew puis d'aller chercher l'executable dans le la librairie installée par Brew selon l'emplacement attitré sur votre ordinateur et de le placer dans le dossier ChessRL/engine à la place du fichier stockfish existant. 
```bash
brew install stockfish
```

Pour linux : nous avons mis à disposition un agent disponible sur le site officiel de Stockfish mais n'avons pu testé la solution sur cet OS.

## Features

### Capture Chess 
- But : Un agent joue aux échecs contre Stockfish ou contre un humain.
- Motivation: Réussir à faire apprendre un agent via du Deep Q-Learning, voir l'impact de différentes méthodes nottament un warmup de notre réseau.
- Concepts: Q-learning, value function approximation, experience replay, fixed-q-targets, policy gradients, warmup.

## Installation
```bash
pip install git+https://github.com/PeopleOfPlay/ChessRL.git
```

