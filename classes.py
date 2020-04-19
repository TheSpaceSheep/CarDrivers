from functions import distance, rotate, rectangle_vertices, is_right_of_line, sec_to_hmsc
from physics import Time, ASPHALT_DRAG, ADH, MAX_SPEED, MAX_SPEED_BACKWARDS
import numpy as np
import time

from gui_manager import world


class Point:
    def __init__(self, x=0, y=0, radius=3, color='white'):
        self.x = x
        self.y = y
        self.r = radius
        self.color = color

    def display(self, tag='debug'):
        world.create_oval(self.x - self.r, self.y - self.r,
                          self.x + self.r, self.y + self.r, fill=self.color, tag=tag)
        world.pack()


class Car:
    def __init__(self, x=0, y=0, heading=0, color='red', width=20, length=35,
                 mass=700, thrust=6, breaks=20, name="car", T=Time(),
                 controls=('Up', 'Left', 'Down', 'Right')):
        self.name = name
        self.x = x
        self.y = y
        self.heading = heading
        self.color = color
        self.width = width
        self.length = length
        self.mass = mass
        self.thrust = thrust
        self.breaks = breaks
        self.speed = 0
        self.T = T
        self.up, self.left, self.down, self.right = controls

                          # -1 means right
        self.turning = 0  # 0  means straight
                          # 1  means left

        self.omega = 0  # rotational speed of a turning car
        self.dtheta = self.omega * self.T.dt  # angle by which a car rotates (at each time step) when turning
        self.leave_trail = None

    def display(self):
        """
       B _________ A       wheels : four small rectangles w on the sides of the rectangle
        |         |        coordinates : w1x, w1y, w2x, w2y, ...
        |    O    |
        |_________|
       C           D
        """
        coords_body = rectangle_vertices(self.x, self.y, width=self.width, length=self.length, angle=self.heading)

        w1x, w1y = rotate(self.x + self.length/4, self.y - self.width/2, self.heading, self.x, self.y)  # front left
        w2x, w2y = rotate(self.x - self.length/3.1, self.y - self.width/2, self.heading, self.x, self.y)  # back left
        w3x, w3y = rotate(self.x + self.length/4, self.y + self.width/2, self.heading, self.x, self.y)  # front right
        w4x, w4y = rotate(self.x - self.length/3.1, self.y + self.width/2, self.heading, self.x, self.y)  # back right

        fwar = self.turning * 10  # front wheels additional rotation (if car is turning)

        coords_wheel_1 = rectangle_vertices(w1x, w1y, width=self.width/4, length=self.length/4, angle=self.heading+fwar)
        coords_wheel_2 = rectangle_vertices(w2x, w2y, width=self.width/4, length=self.length/4, angle=self.heading)
        coords_wheel_3 = rectangle_vertices(w3x, w3y, width=self.width/4, length=self.length/4, angle=self.heading+fwar)
        coords_wheel_4 = rectangle_vertices(w4x, w4y, width=self.width/4, length=self.length/4, angle=self.heading)

        world.create_polygon(coords_body, fill=self.color, outline='black', width=2, tag=self.name)
        world.create_polygon(coords_wheel_1, fill='black', tag=self.name + "_wfl")
        world.create_polygon(coords_wheel_2, fill='black', tag=self.name + "_wbl")
        world.create_polygon(coords_wheel_3, fill='black', tag=self.name + "_wfr")
        world.create_polygon(coords_wheel_4, fill='black', tag=self.name + "_wbr")

        world.pack()

    def erase(self, *args):
        """erase every part of the car except the ones specified in argsi
        wfl = wheel front left
        wbr = wheel bottom right
        etc"""
        if not self.leave_trail == 'shadow':
            if self.name + "_wfl" not in args:
                world.delete(self.name + "_wfl")
            if self.name + "_wbl" not in args:
                world.delete(self.name + "_wbl")
            if self.name + "_wfr" not in args:
                world.delete(self.name + "_wfr")
            if self.name + "_wbr" not in args:
                world.delete(self.name + "_wbr")
            if self.name not in args:
                world.delete(self.name)
        if self.leave_trail == 'point':
            Point(self.x, self.y).display()

    def accelerate(self, acc):
        self.speed += acc * self.T.dt
        if self.speed > MAX_SPEED:
            self.speed = MAX_SPEED
        elif self.speed < -MAX_SPEED_BACKWARDS:
            self.speed = -MAX_SPEED_BACKWARDS

    def turn(self, direction):
        if direction == 'Left':
            self.turning = 1
        elif direction == 'Right':
            self.turning = -1
        else:
            self.turning = 0

    def forwards(self):
        if self.speed < 0:
            self.accelerate(self.breaks)
            if self.speed > 0:
                self.speed == 0
        elif self.speed >= 0:
            self.accelerate(self.thrust)

    def backwards(self):
        if self.speed > 0:
            self.accelerate(-self.breaks)
            if self.speed < 0:
                self.speed = 0
        elif self.speed <= 0:
            self.accelerate(-self.thrust)

    def move(self, drag):
        self.omega = 4_000/(abs(self.speed)+2)
        self.dtheta = self.omega * self.T.dt
        self.heading += self.turning * self.dtheta * self.speed/100
        heading_in_rad = 2*np.pi * self.heading / 360
        self.x += np.cos(heading_in_rad) * self.speed * self.T.dt
        self.y -= np.sin(heading_in_rad) * self.speed * self.T.dt
        if self.speed > 0:
            self.accelerate(- ADH * abs(self.turning) / self.mass)

        self.accelerate(-self.speed * drag / self.mass)

    def trace(self, leave_trail):
        """
        (De)activates trace to keep visual track of where the car has been.
        Arguments can be None, 'shadow' or 'point'."""
        self.leave_trail = leave_trail

    def teleport(self, x, y, heading):
        self.x = x
        self.y = y
        self.heading = heading


class Curve:
    def __init__(self, xa, ya, xb, yb, c):
        """a curve from point a to point b,
        of curvature c (0 for a straight line)
        c > 0 : turns left"""
        self.xa = xa
        self.ya = ya
        self.xb = xb
        self.yb = yb
        self.c = c

    def function(self, t):
        xa, ya = self.xa, self.ya
        xb, yb = self.xb, self.yb

        reverse = (xa < xb and ya > yb) or \
                  (xa > xb and ya < yb)

        if reverse:
            c = -self.c
        else:
            c = self.c

        def f(t):
            if c < 0:
                return (1 - t) * np.exp(c*t)
            else:
                return - t * np.exp(c * (t-1)) + 1

        # to prevent dividing by 0
        if xa == xb:
            xa += 0.1

        return (ya-yb) * f((t-xa)/(xb-xa)) + yb

    def __repr__(self):
        s = "Curve from ({0}, {1}) to ({2}, {3})".format(self.xa, self.ya, self.xb, self.yb)
        return s

    def display(self, step=10, f=None):
        if not f:
            f = self.function
        t = min(self.xa, self.xb)
        while t < max(self.xa, self.xb):
            # Point(t, f(t), radius=5).display(tag='test')
            world.pack()
            world.create_line(t, f(t), t+step, f(t+step), fill='grey', width=5)
            t += step

    def points(self, step, f=None):
        if not f:
            f = self.function
        points = []
        t = min(self.xa, self.xb)
        while t < max(self.xa, self.xb):
            points.append((t, f(t)))
            t += step
        return points

    def is_left(self, x, y):
        if self.xa > self.xb:
            return self.function(x) < y
        else:
            return self.function(x) > y

    def is_right(self, x, y):
        return not self.is_left(x, y)


class Checkpoint:
    def __init__(self, xg, yg, xd, yd):
        self.xg = xg
        self.yg = yg
        self.xd = xd
        self.yd = yd

    def reached(self, car):
        x_milieu = (self.xg + self.xd) / 2
        y_milieu = (self.yg + self.yd) / 2
        width = distance(self.xg, self.yg, self.xd, self.yd)

        angle = np.arccos((self.xd - self.xg)/width)
        if self.yd < self.yg:
            angle = -angle

        ax, ay, bx, by, cx, cy, dx, dy = rectangle_vertices(x_milieu, y_milieu,
                                                            width=car.length,
                                                            length=width + car.width,
                                                            angle=360 * angle / (2 * np.pi))

        if is_right_of_line(car.x, car.y, bx, by, ax, ay) and is_right_of_line(car.x, car.y, cx, cy, bx, by) and \
           is_right_of_line(car.x, car.y, dx, dy, cx, cy) and is_right_of_line(car.x, car.y, ax, ay, dx, dy):
            return True
        else:
            return False


class TrackPiece:
    def __init__(self, x, y, angle=0, width=50, is_checkpoint=True,
                 is_start=False, is_finish=False, nb_checkpoints=10):
        """
        x, y are coordinates (center)
        prev, next are TrackPiece objects
        curv_p, curv_s are parameters (0 if the track goes in a straight line)
        """
        self.x = x
        self.y = y
        self.angle = 2 * np.pi * angle / 360
        self.width = width
        self.is_checkpoint = is_checkpoint
        self.is_start = is_start
        self.is_finish = is_finish
        self.checkpoints = []
        self.nb_checkpoints = nb_checkpoints

        self.highlight_time = 0

    def __repr__(self):
        s = "Track piece object : angle = {0}°, width = {1}".format(int(360*self.angle/(2*np.pi)), self.width)
        if self.is_checkpoint:
            s = "Checkpoint " + s
        s += '\n' + "x : {0}, y : {1}".format(self.x, self.y)
        return s

    def display(self, prec, succ, curv_p=10, curv_s=10, pre_suc=0, highlight=False):
        """display the track between Trackpoints prec, self and succ
        with curvatures curv_p for prec and curv_s for succ

        set pre_suc to -1 to display only prec
                        1 to display only succ


        ag                  cg

        ad         bg
                            cd
                   bd

        prec      self     succ
        """

        bgx = self.x + np.cos(self.angle) * self.width / 2
        bgy = self.y - np.sin(self.angle) * self.width / 2

        bdx = self.x - np.cos(self.angle) * self.width / 2
        bdy = self.y + np.sin(self.angle) * self.width / 2

        if self.is_start:
            world.create_line(bgx, bgy, bdx, bdy, width=5, fill='white')
        elif self.is_finish:
            world.create_line(bgx, bgy, bdx, bdy, width=2, fill='red')
        elif self.is_checkpoint:
            world.create_line(bgx, bgy, bdx, bdy, width=2, fill='blue')

        if pre_suc == 0 or pre_suc == -1:
            agx = prec.x + np.cos(prec.angle) * prec.width / 2
            agy = prec.y - np.sin(prec.angle) * prec.width / 2

            adx = prec.x - np.cos(prec.angle) * prec.width / 2
            ady = prec.y + np.sin(prec.angle) * prec.width / 2

            c_prec_g = Curve(agx, agy, bgx, bgy, curv_p)
            c_prec_d = Curve(adx, ady, bdx, bdy, curv_p)

            c_prec_g.display()
            c_prec_d.display()

        if pre_suc == 0 or pre_suc == 1 and not self.is_finish:
            cgx = succ.x + np.cos(succ.angle) * succ.width / 2
            cgy = succ.y - np.sin(succ.angle) * succ.width / 2

            cdx = succ.x - np.cos(succ.angle) * succ.width / 2
            cdy = succ.y + np.sin(succ.angle) * succ.width / 2

            c_succ_g = Curve(bgx, bgy, cgx, cgy, curv_s)
            c_succ_d = Curve(bdx, bdy, cdx, cdy, curv_s)

            points_g = c_succ_g.points(100)
            points_d = c_succ_d.points(100)
            for g, d in zip(points_g, points_d):
                world.create_line(*g, *d, width=2,
                                      fill='blue')
                self.checkpoints.append(Checkpoint(*g, *d))

            c_succ_g.display()
            c_succ_d.display()

    def highlight(self):
        gx = self.x + np.cos(self.angle) * self.width / 2
        gy = self.y - np.sin(self.angle) * self.width / 2

        dx = self.x - np.cos(self.angle) * self.width / 2
        dy = self.y + np.sin(self.angle) * self.width / 2

        world.delete(str(self.highlight_time))
        self.highlight_time = time.time()
        world.create_line(gx, gy, dx, dy, width=5, fill='yellow', tag=str(self.highlight_time))

    def reached(self, car):
        assert self.is_checkpoint or self.is_finish or self.is_start
        ax, ay, bx, by, cx, cy, dx, dy = rectangle_vertices(self.x, self.y,
                                                            width=car.length,
                                                            length=self.width + car.width,
                                                            angle=360 * self.angle / (2 * np.pi))

        if is_right_of_line(car.x, car.y, bx, by, ax, ay) and is_right_of_line(car.x, car.y, cx, cy, bx, by) and \
           is_right_of_line(car.x, car.y, dx, dy, cx, cy) and is_right_of_line(car.x, car.y, ax, ay, dx, dy):
            return True
        else:
            return False


class Track:
    def __init__(self, pieces, curvatures, is_cyclic=True, nb_laps=1):
        """pieces is a list of TrackPiece obects
           curve is a list of curvature parameters for the curves between pieces"""

        assert pieces and len(pieces) >= 2, "you must at least provide two TrackPiece objects when creating a track"
        assert is_cyclic or (not is_cyclic) == (nb_laps == 1), "you can only have one lap on a non-cyclic track"

        self.pieces = pieces
        self.curvatures = curvatures
        self.is_cyclic = is_cyclic

        self.ai_checkpoints = []

        self.start = pieces[0]
        self.start.is_start = True

        if not is_cyclic:
            self.finish = pieces[-1]
            self.finish.is_finish = True
        else:
            self.finish = None

        self.cars = {}  # keys are car names, values are car objects linked to the track
        self.current_checkpoint = {}  # keys are car names, contains current checkpoint for each car
        self.ai_checkpoint = {}  # same with intermediate checkpoints for ai

        self.times = {}  # for each car, the latest time it passed through the start
        self.best_times = {}  # for each car, the list of best times for each checkpoint
        self.total_time = {}  # for each car, the time it took too complete a whole race

        self.current_lap = {}  # for each car, its current lap
        self.nb_laps = nb_laps

        self.podium = []

        self.curves = []  # [(curve_gauche, curve_droite), ...]
        for i in range(len(pieces) - 1*(not self.is_cyclic)):
            bgx = self.pieces[i].x + np.cos(self.pieces[i].angle) * self.pieces[i].width / 2
            bgy = self.pieces[i].y - np.sin(self.pieces[i].angle) * self.pieces[i].width / 2

            bdx = self.pieces[i].x - np.cos(self.pieces[i].angle) * self.pieces[i].width / 2
            bdy = self.pieces[i].y + np.sin(self.pieces[i].angle) * self.pieces[i].width / 2

            succ = self.pieces[(i + 1) % len(pieces)]
            curv_s = self.curvatures[(i + 1) % len(pieces)]

            cgx = succ.x + np.cos(succ.angle) * succ.width / 2
            cgy = succ.y - np.sin(succ.angle) * succ.width / 2

            cdx = succ.x - np.cos(succ.angle) * succ.width / 2
            cdy = succ.y + np.sin(succ.angle) * succ.width / 2

            # bgx, cgx = min(bgx, cgx), max(bgx, cgx)
            # bgy, cgy = min(bgy, cgy), max(bgy, cgy)

            self.curves.append(
                (Curve(bgx, bgy, cgx, cgy, curv_s), Curve(bdx, bdy, cdx, cdy, curv_s))
            )

    def display(self, text=True):
        for i in range(len(self.pieces)):
            self.pieces[i].display(self.pieces[i-1], self.pieces[(i+1) % len(self.pieces)],
                                   self.curvatures[i-1], self.curvatures[(i+1) % len(self.pieces)], 1)
        for p in self.pieces:
            self.ai_checkpoints += p.checkpoints

    def display_text(self, text_dic):
        x = 150  # position of checkpoint text
        y = 30
        if text_dic is None:
            return

        for key, string in text_dic.items():
            if key == 'countdown':
                world.delete('countdown')
                world.create_text(1800/2, 900/2, fill='green', text=string, font='Sans 100', tag='countdown')

            for name in self.cars:
                if key == name + 'checkpoint_time':
                    world.delete('car_checkpoint')
                    world.create_text(x, y, fill='green', text=string, font="Times 15", tag='car_checkpoint')

                if key == name + 'nb_laps':
                    world.delete('lap_nb')
                    world.create_text(1800/2, 900-40, fill='green', text=string, font="Sans 30", tag='lap_nb')

                if key == name + 'lap_time':
                    world.delete('lap_time')
                    world.create_text(x, y + 30, fill='green', text=string, font="Times 15", tag='lap_time')

                if key == name + 'best_lap':
                    world.delete('best_lap')
                    world.delete('start')
                    world.create_text(x, y + 50, fill='green', text=string, font="Times 15", tag='best_lap')

                if key == name + 'starting_race':
                    world.create_text(1800/2, 900/2+20, fill='red', text=string, font="Times 30", tag='start')

                if key == name + 'end_race':
                    if len(self.podium) == len(self.cars):
                        world.create_text(1800 / 2 - 50, 800 / 2 - 30, fill='green', text=string,
                                          font="Sans 15", tag='end_race')
                        for i in range(len(self.podium)):
                            car = self.podium[i]
                            s = "{0} : {1} {2}h{3}'{4}\"{5}".format(i+1, car.name,
                                                                       *sec_to_hmsc(self.total_time[car.name]))
                            world.create_text(1800 / 2 - 50, 800 / 2 + i*25, fill=self.podium[i].color, text=s,
                                              font="Sans 15", tag='end_race')

    def __repr__(self):
        for piece in self.pieces:
            print(piece)

    def put_car_on_track(self, car):
        self.cars[car.name] = car
        self.current_checkpoint[car.name] = None
        self.ai_checkpoint[car.name] = None
        self.total_time[car.name] = 0
        self.times[car.name] = [-1.] + [1e20] * (len(self.pieces)-1)
        self.best_times[car.name] = [1e20] * len(self.pieces)
        self.current_lap[car.name] = 0

    def end_race(self, car):
        self.current_checkpoint[car.name] = None
        self.ai_checkpoint[car.name] = None
        self.podium.append(car)

    def start_race(self, cars, window, countdown=True):
        world.delete('all')
        self.podium = []
        self.display()
        start_time = time.time()
        pos = 0
        un_sur_deux = 1
        for car in cars:
            self.put_car_on_track(car)

            x = self.start.x - np.sin(self.start.angle) * car.length + np.cos(self.start.angle)*(pos - car.width/2)
            y = self.start.y - np.cos(self.start.angle) * car.length + np.sin(self.start.angle)*(pos - car.width/2)
            pos = pos + car.width * un_sur_deux
            un_sur_deux = - un_sur_deux
            car.teleport(x, y, 360*self.start.angle/(2*np.pi) - 90)
            car.speed = 0
            self.current_checkpoint[car.name] = None
            self.next_checkpoint(car)
            self.next_ai_checkpoint(car)
            self.total_time[car.name] = start_time - 3

            car.display()
        if countdown:
            for i in range(3, 0, -1):
                self.display_text({'countdown': str(i)})
                world.pack()
                window.update()
                time.sleep(1)

            world.delete('countdown')

    def next_checkpoint(self, car):
        if not self.cars.get(car.name):
            self.put_car_on_track(car)

        if not self.current_checkpoint[car.name]:
            i = 0
        else:
            i = self.pieces.index(self.current_checkpoint[car.name]) + 1
            i %= len(self.pieces)
            while not self.pieces[i].is_checkpoint and \
                  not self.pieces[i].is_start      and \
                  not self.pieces[i].is_finish:
                i += 1
                i %= len(self.pieces)

        self.current_checkpoint[car.name] = self.pieces[i]

        return self.pieces[i]

    def next_ai_checkpoint(self, car):
        if not self.cars.get(car.name):
            self.put_car_on_track(car)

        if not self.ai_checkpoint[car.name]:
            i = 0
        else:
            i = self.ai_checkpoints.index(self.ai_checkpoint[car.name])

        self.ai_checkpoint[car.name] = self.ai_checkpoints[(i+1)%len(self.ai_checkpoints)]


    def update(self):
        output = {}
        tmp = time.time()
        for c in self.pieces:
            if tmp - c.highlight_time > 0.2:
                world.delete(str(c.highlight_time))


        for name in self.cars:
            car = self.cars[name]

            output['reward' + car.name] = 0

            ai_checkpoint = self.ai_checkpoint[car.name]
            if ai_checkpoint is not None and ai_checkpoint.reached(car):
                self.next_ai_checkpoint(car)
                output['reward' + car.name] = 1

            checkpoint = self.current_checkpoint[name]
            if checkpoint is not None and checkpoint.reached(car):
                output['reward' + car.name] = 1
                self.next_checkpoint(car)
                checkpoint.highlight()

                if checkpoint == self.start:
                    self.current_lap[name] += 1
                    output[name + 'nb_laps'] = "Lap : {}/{}".format(min(self.current_lap[name], self.nb_laps),
                                                                    self.nb_laps)
                    if self.times[car.name][0] != -1:
                        lap_time = "lap time : {0}h {1}'{2}\"{3}\"'".format(*sec_to_hmsc(tmp - self.times[name][0]))
                        output[name + 'lap_time'] = lap_time
                        print(lap_time)

                        if self.best_times[car.name][0] > tmp - self.times[name][0]:
                            self.best_times[car.name][0] = tmp - self.times[name][0]
                            best_lap = "best lap : {0}h {1}'{2}\"{3}\"'".format(*sec_to_hmsc(self.best_times[name][0]))
                            output[name + 'best_lap'] = best_lap
                            print(best_lap)

                    else:
                        print("starting race !")
                        output[name + 'starting_race'] = 'starting race !'

                    if self.current_lap[name] - 1 == self.nb_laps:
                        self.total_time[car.name] = tmp - self.total_time[car.name]
                        self.end_race(car)
                        output[name + 'end_race'] = "End of race !"

                    self.times[name][0] = tmp
                else:
                    self.times[name][self.pieces.index(checkpoint)] = tmp
                    checkpoint_time = "checkpoint time : {0}h {1}'{2}\"{3}".format(*sec_to_hmsc(tmp - self.times[name][0]))
                    print(checkpoint_time)
                    output[name + 'checkpoint_time'] = checkpoint_time
        return output


def is_out(track, car=None, x=None, y=None):
    if car is not None:
        x = car.x
        y = car.y

    else:
        assert x is not None and y is not None

    for i in range(len(track.curves)):
        c_g, c_d = track.curves[i]
        ok = (min(c_g.xa, c_g.xb) <= x <= max(c_g.xa, c_g.xb) and
              min(c_g.ya, c_g.yb) <= y <= max(c_g.ya, c_g.yb)) or \
             (min(c_d.xa, c_d.xb) <= x <= max(c_d.xa, c_d.xb) and
              min(c_d.ya, c_d.yb) <= y <= max(c_d.ya, c_d.yb))

        if c_g.is_right(x, y) and c_d.is_left(x, y) and ok:
            return False
    return True


def closest_point(car, track, angle):
    """return the approximate coordinates of the closest point to the car along
    the direction given by theta (relative to the car's heading)"""
    x_inf, y_inf = car.x, car.y

    if not is_out(track, car):
        t = 100
        x, y = rotate(car.x + t, car.y, car.heading + angle, car.x, car.y)
        while not is_out(track, x=x, y=y):
            t *= 2
            x, y = rotate(car.x + t, car.y, car.heading + angle, car.x, car.y)

        epsilon = 10
        while epsilon > 5:
            x_tmp, y_tmp = (x + x_inf) / 2, (y + y_inf) / 2

            if not is_out(track, x=x_tmp, y=y_tmp):
                x_inf, y_inf = x_tmp, y_tmp
            else:
                x, y = x_tmp, y_tmp

            epsilon = distance(x, y, x_inf, y_inf)
    else:
        t = 5
        x, y = rotate(car.x + t, car.y, car.heading + angle, car.x, car.y)
        while is_out(track, x=x, y=y) and t < 300:
            t += 25
            x, y = rotate(car.x + t, car.y, car.heading + angle, car.x, car.y)

        epsilon = 10
        while epsilon > 5:
            x_tmp, y_tmp = (x + x_inf) / 2, (y + y_inf) / 2

            if is_out(track, x=x_tmp, y=y_tmp):
                x_inf, y_inf = x_tmp, y_tmp
            else:
                x, y = x_tmp, y_tmp

            epsilon = distance(x, y, x_inf, y_inf)

    return x, y




def distances(car, track, nb_angles=8, debugging=False):
    if debugging and world is not None:
        world.delete('debug')
    dists = []
    for i in range(nb_angles):
        theta = -180 + i*360/(nb_angles)
        x, y = closest_point(car, track, theta)
        dists.append(distance(car.x, car.y, x, y))
        if debugging and world is not None:
            world.create_line(car.x, car.y, x, y, fill='green', tag='debug')
    return dists




