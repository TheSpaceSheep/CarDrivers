ASPHALT_DRAG = 20
SAND_DRAG = 500
ADH = 500
MAX_SPEED = 200
MAX_SPEED_BACKWARDS = 40


class Time:
    def __init__(self, dt=0.04, dilation=1):
        self.dt = dt
        self.dilation = dilation

    def dilate(self, dilation=1):
        self.dilation = dilation
