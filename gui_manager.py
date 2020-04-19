import tkinter as tk
import os
import hyperparameters as hp


Fenetre = tk.Tk()
world = tk.Canvas(Fenetre, width=hp.WIDTH, height=hp.HEIGHT)
world.pack()

close_flag = False

# --------------------------- EVENT HANDLING --------------------------

history = []

def on_closing():
    global close_flag
    close_flag = True
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
