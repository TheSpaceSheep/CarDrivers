# CarDrivers

This game engine developed entirely with tkinter and numpy can be played as is (alone or multiplayer) or can be used as an environment for agents (has no Open AI Gym wrapping yet).

There are two available agents : 
* a supervised agent that takes input data from the human player and tries to approximate it using a simple neural net.
* a reinforcement learning agent that uses a Deep Q Network (implemented following this [tutorial](https://deeplizard.com/learn/video/PyQNfsGUnQA) ).

Current Status : The supervised agent does not learn effectively. After a few thousand episodes, the RL agent learns to complete the full track, but this result is highly dependant on hyperparameters.

