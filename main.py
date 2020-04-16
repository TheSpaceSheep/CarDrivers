import tkinter as tk
import numpy as np
from classes import Point, Car, Curve, TrackPiece, Track, is_out, closest_point, distances
from functions import rotate
from physics import Time, ASPHALT_DRAG, SAND_DRAG
from rl_agent import RL_Driver
import hyperparameters as hp
import os

os.system("xset r off")  # disable key repeat

Fenetre = tk.Tk()
WIDTH = 1800
HEIGHT = 900
world = tk.Canvas(Fenetre, width=WIDTH, height=HEIGHT)
world.pack()

T = Time(dt=0.08, dilation=1)

voiture2 = Car(0, 0, heading=0, length=20, width=10, name="bob", T=T, controls=('z', 'q', 's', 'd'))
voiture2.trace(None)

voiture = Car(0, 0, heading=0, length=20, width=10, name="kloz", T=T, color='pink',
              controls=('Up', 'Left', 'Down', 'Right'))
voiture.trace(None)

cars = [voiture, voiture2]

# -------------------------------------------- TRACK CONSTRUCTION  -----------------------------------------------------
"""track_points = [
    TrackPiece(WIDTH/2, 50, angle=90, width=80),
    TrackPiece(WIDTH-100, HEIGHT/2, angle=0, width=80),
    TrackPiece(5*WIDTH/6, HEIGHT-50, angle=-90, width=80),
    TrackPiece(4*WIDTH/6, HEIGHT - 200, angle = -180, width=80),
    TrackPiece(WIDTH/2, HEIGHT - 400, angle=-90, width=80),
    TrackPiece(2*WIDTH/6, HEIGHT - 200, angle=0, width=80),
    TrackPiece(WIDTH/6, HEIGHT-50, angle=-90, width=80),
    TrackPiece(100, HEIGHT/2, angle=180, width=80)
]"""

track_points = [
    TrackPiece(WIDTH/2, 50, angle=90, width=80),

    TrackPiece(WIDTH -100, HEIGHT/6, angle=0, width=80),
    TrackPiece(WIDTH - 200, 2 * HEIGHT / 6, angle=-90, width=80),
    TrackPiece(WIDTH - 400, HEIGHT / 2, angle=0, width=80),
    TrackPiece(WIDTH - 200, 4 * HEIGHT / 6, angle=90, width=80),
    TrackPiece(WIDTH - 100, 5 * HEIGHT / 6, angle=0, width=80),

    TrackPiece(5*WIDTH/6, HEIGHT-50, angle=-90, width=80),
    TrackPiece(4*WIDTH/6, HEIGHT - 200, angle = -180, width=80),
    TrackPiece(WIDTH/2, HEIGHT - 400, angle=-90, width=80),
    TrackPiece(2*WIDTH/6, HEIGHT - 200, angle=0, width=80),
    TrackPiece(WIDTH/6, HEIGHT-50, angle=-90, width=80),

    TrackPiece(100, 5 * HEIGHT / 6, angle=180, width=80),
    TrackPiece(200, 4 * HEIGHT / 6, angle=90, width=80),
    TrackPiece(400, HEIGHT / 2, angle=180, width=80),
    TrackPiece(200, 2 * HEIGHT / 6, angle=-90, width=80),
    TrackPiece(100, HEIGHT / 6, angle=180, width=80),
]

curvatures = [15, 15, 5, -5, -5, 5, 15, 15, -5, -5, 5, 15, 15, -5, -5, 5, 15]

Piste = Track(track_points, curvatures, nb_laps=10)
Piste.display(world)
# ----------------------------------------------------------------------------------------------------------------------

# Piste.put_car_on_track(voiture)
Piste.put_car_on_track(voiture2)

smith = RL_Driver(voiture2)
smith.initiate(Piste)

# ------------------------------------------------- EVENT HANDLING -----------------------------------------------------
history = []


def on_closing():
    os.system("xset r on")
    Fenetre.destroy()


def keyup(e):
    if e.keysym in history:
        history.pop(history.index(e.keysym))

    for car in cars:
        if e.keysym in [car.left, car.right]:
            car.turn("straight")

    if e.keysym == 'h':
        T.dilate(1)


def keydown(e):
    return # to disable key inputs
    if e.keysym not in history:
        history.append(e.keysym)


def restart(e):
    if e.keysm not in history:
        history.append(e.keysym)


Fenetre.bind("<Key>", keydown)
Fenetre.bind("<KeyRelease>", keyup)

Fenetre.protocol("WM_DELETE_WINDOW", on_closing)

# ----------------------------------------------------------------------------------------------------------------------
episode_length = 100
while True:
    voiture.erase(world)
    voiture2.erase(world)

    # print(smith.steps_done)
    state = smith.sensors()
    action = smith.decision()
    distances(smith.car, Piste, world=world)

    # ai presses on keys
    ai_keys = []
    if action in (0, 1, 7):
        ai_keys.append(voiture2.up)
    if action in (1, 2, 3):
        ai_keys.append(voiture2.right)
    if action in (3, 4, 5):
        ai_keys.append(voiture2.down)
    if action in (5, 6, 7):
        ai_keys.append(voiture2.left)

    print(ai_keys)
    for keysym in voiture2.up, voiture2.right, voiture2.down, voiture2.left:
        if keysym in history:
            history.remove(keysym)

    for keysym in ai_keys:
        if keysym not in history:
            history.append(keysym)

    for keysym in history:
        for car in cars:
            if keysym == car.up:
                car.forwards()
            elif keysym == car.down:
                car.backwards()
            elif keysym == car.left:
                car.turn('Left')
            elif keysym == car.right:
                car.turn('Right')
            elif car.right not in history and car.left not in history:
                car.turn("straight")

        if keysym == 'h':
            T.dilate(T.dilation+0.1)
        elif keysym == 'BackSpace':
            Piste.start_race([voiture, voiture2], world, Fenetre)
            print(Piste.current_checkpoint)
            #smith.initiate(Piste, next_check=True)
            #smith.learning = True

    screen_output = Piste.update(world)
    Piste.display_text(world, screen_output)

    if is_out(Piste, voiture):
        voiture.move(SAND_DRAG)
    else:
        voiture.move(ASPHALT_DRAG)

    if is_out(Piste, voiture2):
        voiture2.move(SAND_DRAG)
    else:
        voiture2.move(ASPHALT_DRAG)

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
    if is_out(Piste, smith.car) or smith.steps_done % episode_length == 0:
        Piste.start_race([voiture, voiture2], world, Fenetre, countdown=False)
        episode_length = round(100 + (10000 - 100)*(1-np.exp(-smith.steps_done*0.0001)))

    voiture.display(world)
    voiture2.display(world)
    Fenetre.update()
    world.after(int(T.dilation * T.dt * 200))
