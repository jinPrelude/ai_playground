import krpc
import time
import numpy as np

class hover_v0:
    def __init__(self, sas=True, max_altitude = 1000, max_step=100):
        self.conn = krpc.connect(name='hover')
        self.vessel = self.conn.space_center.active_vessel
        self.step_count = 0
        self.done = False
        self.reward = 0
        self.max_altitude = max_altitude

        self.observation_space = 3 # thrust, current_mass
        self.action_space = 1 # throttle ratio
        self.action_max = 1.
        self.action_min = 0.1

        self.initial_throttle = 0.3

        self.sas = sas
        self.target_altitude = 100
        self.max_step = max_step


    def reset(self):

        # Quicksave initial state
        self.done = False
        self.conn.space_center.quicksave()

        # Initialize sas
        self.vessel.control.sas = self.sas

        # Initialize throttle
        self.vessel.control.throttle = self.initial_throttle

        # Set target altitude
        self.target_altitude = np.random.randint(100, self.max_altitude)

        print('Target altitude : ', self.target_altitude)
        self.step_count = 0

        self.reward = 0

        self.vessel.control.activate_next_stage()\

        return (self.vessel.thrust, self.vessel.mass, self.target_altitude)

    def step(self, action):
        self.decision(action)
        if self.step_count >= self.max_step :
            self.done = True
            self.step_count = 0
            self.conn.space_center.quickload()
        else :
            self.step_count += 1
            self.reward = -abs(self.vessel.flight().mean_altitude - self.target_altitude)
        time.sleep(0.085)
        return (self.vessel.thrust, self.vessel.mass, self.target_altitude), self.reward, self.done

    def decision(self, action):
        self.vessel.control.throttle = float(action[0])

class hover_v1:
    def __init__(self, sas=True, max_altitude = 1000, max_step=100):
        self.conn = krpc.connect(name='hover')
        self.vessel = self.conn.space_center.active_vessel
        self.step_count = 0
        self.done = False
        self.reward = 0
        self.max_altitude = max_altitude

        self.observation_space = 4 # thrust, current_mass
        self.action_space = 1 # throttle ratio
        self.action_max = 1.
        self.action_min = 0.

        self.initial_throttle = 0.3

        self.sas = sas
        self.target_altitude = 100
        self.max_step = max_step


    def reset(self):

        # Quicksave initial state
        self.done = False
        self.conn.space_center.quicksave()

        # Initialize sas
        self.vessel.control.sas = self.sas

        # Initialize throttle
        self.vessel.control.throttle = self.initial_throttle

        # Set target altitude
        self.target_altitude = np.random.randint(100, self.max_altitude)

        print('Target altitude : ', self.target_altitude)
        self.step_count = 0

        self.reward = 0

        self.vessel.control.activate_next_stage()\

        return (self.vessel.thrust, self.vessel.mass, self.target_altitude, self.vessel.flight().mean_altitude)

    def step(self, action):
        self.decision(action)
        if self.step_count >= self.max_step :
            self.done = True
            self.conn.space_center.quickload()
        else :
            self.step_count += 1
            self.reward = -abs(self.vessel.flight().mean_altitude - self.target_altitude)
        time.sleep(0.08)
        return (self.vessel.thrust, self.vessel.mass, self.target_altitude, self.vessel.flight().mean_altitude), self.reward, self.done

    def decision(self, action):
        self.vessel.control.throttle = float(action)