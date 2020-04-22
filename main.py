import numpy as np
import torch
import os
os.system("xset r off")  # disable key repeat

from classes import Point, Car, Curve, TrackPiece, Track, is_out, closest_point, distances
from functions import rotate
from physics import Time, ASPHALT_DRAG, SAND_DRAG
from rl_agent import RL_Driver
import hyperparameters as hp
import gui_manager as gui
from track_construction import track_points1, curvatures1


T = Time(dt=0.042, dilation=1)

voiture1 = Car(0, 0, heading=0, length=20, width=10, name="kloz", T=T, color='pink',
             controls=('Up', 'Left', 'Down', 'Right'))
voiture1.trace(None)

voiture2 = Car(0, 0, heading=0, length=20, width=10, name="bob", T=T,
               controls=('z', 'q', 's', 'd'), thrust=50)
voiture2.trace(None)

cars = [voiture1, voiture2]
# cars = [voiture2]

# ----------------------------- TRACK CONSTRUCTION  ------------------------------

Piste = Track(track_points1, curvatures1, nb_laps=10)
Piste.display()

for car in cars:
    Piste.put_car_on_track(car)

smith = RL_Driver(voiture2)
smith.initiate(Piste)

# ---------------------------------- MAIN LOOP ----------------------------------

episode_length = 1000  # upper bound for the number of steps in each episode
while not gui.close_flag:
    # reinitializing car graphics an keypresses
    for car in cars:
        car.erase()


    state = smith.sensors()
    print(smith.episode_steps)
    # agent must wait 12 steps before taking new action (to make it more
    # human-like)
    action =  smith.decision(state)
    ai_keys =  smith.action_to_keys(action)

    for keysym in voiture2.up, voiture2.right, voiture2.down, voiture2.left:
        if keysym in gui.history:
            gui.history.remove(keysym)
    for keysym in ai_keys:
        if keysym not in gui.history:
            gui.history.append(keysym)

    print(gui.history)
    # processing keystrokes
    for car in cars:
        if car.right not in gui.history and car.left not in gui.history:
            car.turn('straight')
        for keysym in gui.history:
            if keysym == car.up:
                car.forwards()
            elif keysym == car.down:
                car.backwards()
            elif keysym == car.left:
                car.turn('Left')
            elif keysym == car.right:
                car.turn('Right')

            if keysym == 'h':
                T.dilate(T.dilation+0.1)
            elif keysym == 'BackSpace':
                Piste.start_race(cars, gui.Fenetre)
                print(Piste.current_checkpoint)

            if "-UP" in keysym:
                for car in cars:
                    if keysym[:-3] in [car.left, car.right]:
                        car.turn("straight")

                if keysym[:-3] == 'h':
                    T.dilate(1)
                gui.history.pop(gui.history.index(keysym))


    screen_output = Piste.update()
    Piste.display_text(screen_output)

    for car in cars:
        if is_out(Piste, car):
            car.move(SAND_DRAG)
        else:
            car.move(ASPHALT_DRAG)


    print(smith.steps_done, '\t', end='')
    if is_out(Piste, smith.car) or smith.episode_steps % episode_length == 0:
        reward = -1 if is_out(Piste, smith.car) else 0
        next_state = torch.zeros(1 + hp.NB_SENSORS).to(hp.DEVICE)

        Piste.start_race(cars, gui.Fenetre, countdown=False)
        episode_length += 100
        smith.new_episode()

    else:
        if 'reward' + voiture2.name in screen_output:
            reward = screen_output['reward' + voiture2.name]
        else:
            reward = 0
        next_state = smith.sensors()

    if reward != 0:
        print("Getting Reward : ", reward)
    smith.memory.push(state, action, reward, next_state)
    smith.episode_steps += 1

    smith.train()

    for car in cars:
        car.display()

    gui.Fenetre.update()
    gui.world.after(int(T.dilation * T.dt * 1000))

