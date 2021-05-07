# CarDrivers

This game engine developed entirely with tkinter and numpy can be played as is (alone or multiplayer) or can be used as an environment for agents.

There are two available agents : 
* a supervised agent that takes input data from the human player and tries to approximate it using a simple neural net.
* a reinforcement learning agent that uses a Deep Q Network (implemented following this [tutorial](https://deeplizard.com/learn/video/PyQNfsGUnQA) ).

Current Status : The supervised agent does not learn effectively. After a few thousand episodes, the RL agent learns to complete the full track, but this result is highly dependant on hyperparameters.

![rl_car_1](https://user-images.githubusercontent.com/30984054/117488393-a4bb5000-af6c-11eb-9975-a2401d1063c8.png)
