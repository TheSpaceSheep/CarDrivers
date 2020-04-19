import numpy as np
import os
os.system("xset r off")  # disable key repeat

from classes import Point, Car, Curve, TrackPiece, Track, is_out, closest_point, distances
from functions import rotate
from physics import Time, ASPHALT_DRAG, SAND_DRAG
from rl_agent import RL_Driver
import hyperparameters as hp
import gui_manager as gui


T = Time(dt=0.08, dilation=1)

voiture2 = Car(0, 0, heading=0, length=20, width=10, name="bob", T=T, controls=('z', 'q', 's', 'd'))
voiture2.trace(None)

# voiture1 = Car(0, 0, heading=0, length=20, width=10, name="kloz", T=T, color='pink',
#             controls=('Up', 'Left', 'Down', 'Right'))
# voiture1.trace(None)

# cars = [voiture1, voiture2]
cars = [voiture2]

# -------------------------------------------- TRACK CONSTRUCTION  -----------------------------------------------------
"""track_points = [
    TrackPiece(hp.WIDTH/2, 50, angle=90, width=80),
    TrackPiece(hp.WIDTH-100, hp.HEIGHT/2, angle=0, width=80),
    TrackPiece(5*hp.WIDTH/6, hp.HEIGHT-50, angle=-90, width=80),
    TrackPiece(4*hp.WIDTH/6, hp.HEIGHT - 200, angle = -180, width=80),
    TrackPiece(hp.WIDTH/2, hp.HEIGHT - 400, angle=-90, width=80),
    TrackPiece(2*hp.WIDTH/6, hp.HEIGHT - 200, angle=0, width=80),
    TrackPiece(hp.WIDTH/6, hp.HEIGHT-50, angle=-90, width=80),
    TrackPiece(100, hp.HEIGHT/2, angle=180, width=80)
]"""

track_points = [
    TrackPiece(hp.WIDTH/2, 50, angle=90, width=80),

    TrackPiece(hp.WIDTH -100, hp.HEIGHT/6, angle=0, width=80),
    TrackPiece(hp.WIDTH - 200, 2 * hp.HEIGHT / 6, angle=-90, width=80),
    TrackPiece(hp.WIDTH - 400, hp.HEIGHT / 2, angle=0, width=80),
    TrackPiece(hp.WIDTH - 200, 4 * hp.HEIGHT / 6, angle=90, width=80),
    TrackPiece(hp.WIDTH - 100, 5 * hp.HEIGHT / 6, angle=0, width=80),

    TrackPiece(5*hp.WIDTH/6, hp.HEIGHT-50, angle=-90, width=80),
    TrackPiece(4*hp.WIDTH/6, hp.HEIGHT - 200, angle = -180, width=80),
    TrackPiece(hp.WIDTH/2, hp.HEIGHT - 400, angle=-90, width=80),
    TrackPiece(2*hp.WIDTH/6, hp.HEIGHT - 200, angle=0, width=80),
    TrackPiece(hp.WIDTH/6, hp.HEIGHT-50, angle=-90, width=80),

    TrackPiece(100, 5 * hp.HEIGHT / 6, angle=180, width=80),
    TrackPiece(200, 4 * hp.HEIGHT / 6, angle=90, width=80),
    TrackPiece(400, hp.HEIGHT / 2, angle=180, width=80),
    TrackPiece(200, 2 * hp.HEIGHT / 6, angle=-90, width=80),
    TrackPiece(100, hp.HEIGHT / 6, angle=180, width=80),
]

curvatures = [15, 15, 5, -5, -5, 5, 15, 15, -5, -5, 5, 15, 15, -5, -5, 5, 15]

Piste = Track(track_points, curvatures, nb_laps=10)
Piste.display()

# Piste.put_car_on_track(voiture1)
Piste.put_car_on_track(voiture2)

smith = RL_Driver(voiture2)
smith.initiate(Piste)

# ---------------------------------- MAIN LOOP  ----------------------------------

episode_length = 100
while not gui.close_flag:
    # reinitializing car graphics and keypresses
    for car in cars:
        car.erase()
        for keysym in car.up, car.right, car.down, car.left:
            if keysym in gui.history:
                gui.history.remove(keysym)

    state = smith.sensors()
    action = smith.decision(state)

    ai_keys = smith.action_to_keys(action)
    for keysym in ai_keys:
        if keysym not in gui.history:
            gui.history.append(keysym)

    for keysym in gui.history:
        for car in cars:
            if keysym == car.up:
                car.forwards()
            elif keysym == car.down:
                car.backwards()
            elif keysym == car.left:
                car.turn('Left')
            elif keysym == car.right:
                car.turn('Right')
            elif car.right not in gui.history and car.left not in gui.history:
                car.turn("straight")

        if keysym == 'h':
            T.dilate(T.dilation+0.1)
        elif keysym == 'BackSpace':
            Piste.start_race(cars, gui.Fenetre)
            print(Piste.current_checkpoint)

    screen_output = Piste.update()
    Piste.display_text(screen_output)

    for car in cars:
        if is_out(Piste, car):
            car.move(SAND_DRAG)
        else:
            car.move(ASPHALT_DRAG)

    if 'reward' + voiture2.name in screen_output:
        reward = 1 + screen_output['reward' + voiture2.name]
    else:
        reward = 0
    print(reward)
    next_state = smith.sensors()
    smith.memory.push(state, action, reward, next_state)
    smith.train()

    print(smith.steps_done)
    print(episode_length)
    if is_out(Piste, smith.car) or smith.episode_steps % episode_length == 0:
        Piste.start_race(cars, gui.Fenetre, countdown=False)
        episode_length = round(100 + (10000 - 100)*(1-np.exp(-smith.steps_done*0.0001)))

    for car in cars:
        car.display()

    gui.Fenetre.update()
    gui.world.after(int(T.dilation * T.dt * 200))

