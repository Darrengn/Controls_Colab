# ------------------------------------------------------------------
# PyBullet Simulation
#
# Function:
# Initialize the simulation, control the robot to collect data, and
# return the dataset.
#
# This class contains a class Simulation that sets up the PyBullet simulation
# for the vehicle and another class PathSimulator that provides a pre-defined
# path and a controller for the operation.
# ------------------------------------------------------------------

from google.colab import output

from pyrc3d.agent import Car
from pyrc3d.simulation import Sim
from pyrc3d.sensors import Lidar
from utilities.timings import Timings
from PID_controller import PID

import numpy as np
from math import *
import matplotlib.pyplot as plt
from IPython.display import clear_output
from time import time
from util import World2Grid, Grid2World

######### This section to load and store the simulation configuration #########

# Declare user-specific paths to files.
ENV_PATH = "/content/Controls_Colab/PID_Colab/configs/env/simple_env.yaml"
CAR_PATH = "/content/Controls_Colab/PID_Colab/configs/car/car_config.yaml"
CAR_URDF_PATH = "/content/Controls_Colab/PID_Colab/configs/resources/f10_racecar/racecar_differential.urdf"

# Constants.
SIMULATE_LIDAR = True

# FPS constants.
PATH_SIM_FPS = 60 # Perform control at 90Hz
LIDAR_FPS = 30 # Simulate lidar at 30Hz
PRINT_FPS = 0.1 # Print `dist` every 10 seconds for debugging
COLLECT_DATA_FPS = 2 # Collect data frequency
IMAGE_FPS = 1

# Declare sensors.
SENSORS = [Lidar]

# Load car and sensor configurations
RAY_LENGTH = 2.0 # length of each ray
RAY_COUNT = 50 # number of laser ray
RAY_START_ANG = 45 # angle b/w robot and the 1st ray
RAY_END_ANG = 135 # angle b/w robot and the last ray
###############################################################################

class Simulation():
    """
    A class used to perform the simulation of environment and robot.

    Attributes:
        sim (Sim): Simulation of environment.
        beta (float): Angular width of each beam.
        rayStartAng (float): Relative angle between robot and the 1st ray.
        Z_max (float): Maximum measurement range of lidar.
    """

    def __init__(self):
        """
        Constructor of Simulation to initialize a simulation.

        Parameters:
            None
        """

        self.sim = Sim(time_step_freq=120, debug=True)

        # Get the angle b/w the 1st and last laser beam
        rayAngleRange = (RAY_END_ANG - RAY_START_ANG) * (pi/180)
        # Angular width of each beam
        self.beta = rayAngleRange/(RAY_COUNT-1)
        # Relative angle between robot and the 1st ray
        self.rayStartAng = RAY_START_ANG * (pi/180)
        # Maximum range of each ray
        self.Z_max = RAY_LENGTH

        # Initialize environment
        planned_path = np.load('/content/Controls_Colab/PID_Colab/trajectory.npy')
        planned_path = [(x, y) for x, y in planned_path]

        planned_path = [
            [0.0, -1.95], [-1.95, -1.95], [-1.95, 1.95], [1.95, 1.95],
            [1.95, -1.95], [0.0, -1.95], [0.0, 1.95], [-1.95, 1.95],
            [-1.95, -1.95], [1.95, -1.95], [1.95, 1.95], [0.0, 1.95],
            [0.0, 0.0]
        ]

        map_size = 5.0
        res = 0.01
        grid_size = int(map_size/res)
        planned_path_grid = [World2Grid(pt, map_size, grid_size, res) for pt in planned_path]
        planned_path_grid = [(x, 500-y) for x, y in planned_path_grid]

        self.sim.create_env(
            env_config=ENV_PATH,
            GUI=False,
            planned_path=planned_path_grid
        )

        # Set simulation response time
        self.path_sim_time = Timings(PATH_SIM_FPS)
        self.lidar_time = Timings(LIDAR_FPS)
        self.print_frequency = Timings(PRINT_FPS)
        self.collect_data_time = Timings(COLLECT_DATA_FPS)
        self.capture_image_time = Timings(IMAGE_FPS)

        # Initial the car
        self.car = Car(
            urdf_path=CAR_URDF_PATH,
            car_config=CAR_PATH,
            sensors=SENSORS
        )

        #starting_point = Grid2World((planned_path[0][0], 500 - planned_path[0][1]), 5.0, 500, 0.01)

        # print('Starting point:', starting_point)

        self.car.place_car(
            self.sim.floor,
            xy_coord=planned_path[0],
        )

        # Initialize path simulator
        self.path_sim = PathSimulator(self.car, PATH_SIM_FPS)

        # self.dataset = {}
        # self.t = 0


    def collectData(self, outputImage):
        """
        The function to collect and store data while running the simulation.

        Parameters:
            None

        Returns:
            None
        """

        image = None

        # Get sensors' data: array of hit points (x, y) in world coord
        rays_data, dists, hitPoints = self.car.get_sensor_data(
            sensor = 'lidar',
            common = False)

        # Obtain the car's current pose and sensor data
        x, y, yaw = self.car.get_state(to_array=False)
        dataset = ((x, y, yaw), hitPoints)

        # if self.print_frequency.update_time():
        #     print("Current pose [x, y, theta]:", (round(x,2), round(y,2), round(yaw,2)))

        # Simulate LiDAR
        if self.lidar_time.update_time():
            self.car.simulate_sensor('lidar', rays_data)

        # Perform car's movement
        if self.path_sim_time.update_time():
            vel, steering = self.path_sim.navigate(x, y, yaw)

            if vel == float('inf'):
                print('Arrived destination.')
                if outputImage:
                    image = self.sim.image_env()
                self.sim.kill_env()
                return image, dataset, -1

            # Perform action
            self.car.act(vel, steering)

            # Advance one time step in the simulation.
            self.sim.step()

        # Capture image of true map
        if outputImage and self.capture_image_time.update_time():
            image = self.sim.image_env()

        return image, dataset, 1

class PathSimulator():
    def __init__(
            self,
            car,
            sim_fps,
        ):

        self.car = car
        self.sim_fps = sim_fps
        self.max_velocity = 60.0  # Maximum velocity of 60 units per second

        self.velocity = 0
        self.steering = 0
        self.pid = PID()

        self.planned_path = np.load('/content/Controls_Colab/PID_Colab/trajectory.npy')
        self.planned_path = [(x, y) for x, y in self.planned_path]

        # Add heading to path, by calculating the angle between two consecutive points
        heading = []
        for i in range(len(self.planned_path)-1):
            x1, y1 = self.planned_path[i]
            x2, y2 = self.planned_path[i+1]
            angle = atan2(y2-y1, x2-x1)
            heading.append(angle)

        heading.append(heading[-1])
        self.planned_path = [(x, y, h) for (x, y), h in zip(self.planned_path, heading)]

        # move: 1: forward, -1: backward, 0: stop
        # turn: 1: right,   -1: left
        # (x, y, heading, turn, move)
        self.waypoints = {
            1: (0.0, -1.95, -pi/2, 1, 1),  2: (-1.95, -1.95, pi, 1, 1),
            3: (-1.95, 1.95, pi/2, 1, 1), 4: (1.95, 1.95, 0.0, 1, 1),
            5: (1.95, -1.95, -pi/2, 1, 1), 6: (0.0, -1.95, pi, 1, 1),
            7: (0.0, 1.95, pi/2, 1, 1), 8: (-1.95, 1.95, pi, -1, 1),
            9: (-1.95, -1.95, -pi/2, -1, 1), 10: (1.95, -1.95, 0.0, -1, 1),
            11: (1.95, 1.95, pi/2, -1, 1), 12: (0.0, 1.95, pi, -1, 1),
            13: (0.0, 0.0, -pi/2, -1, 1),
            # 1: (0.0, 1.95, pi/2, -1, 1), 2: (-1.95, 1.95, pi, -1, 1),
            # 3: (-1.95, -1.95, -pi/2, -1, 1), 4: (1.95, -1.95, 0.0, -1, 1),
            # 5: (1.95, 1.95, pi/2, -1, 1), 6: (0.0, 1.95, pi, -1, 1),
            # 7: (0.0, 0.0, -pi/2, -1, 1),
        }

        # self.waypoints = {}

        # for i, (x, y, heading) in enumerate(self.planned_path):
        #     self.waypoints[i+1] = (x, y, -pi/2, 1, 1)

        self.next = 1 # waypoint number
        self.length = len(self.waypoints)
        self.dist2next = 0
        self.ifTurn = False

    def navigate(self, x, y, yaw):
        if self.next > len(self.waypoints):
            return float('inf'), float('inf')

        next_x, next_y, heading, turn, move = self.waypoints[self.next]
        self.dist2next = np.linalg.norm(
            np.array((next_x, next_y)) - np.array((x, y)))

        # Turn
        if self.ifTurn == False:
            if turn == 0:
                self.ifTurn = True
                return 0.0, 0.0

            if abs(heading - yaw) <= 0.15:
                self.ifTurn = True
                self.steering = 0.0
                return 0.0, 0.0

            if turn == -1: # left
                self.steering = -1.0
            elif turn == 1: # right
                self.steering = 1.0

            return 15.0, self.steering

        # Move
        if self.dist2next >= 0.2:
            self.setVel(move)
            self.setSteer(yaw, heading)
            return self.velocity, self.steering
        else:
            self.next += 1
            print('Moving to next waypoint [', self.next-1, '/', self.length, '].')
            self.ifTurn = False
            self.velocity, self.steering = 0.0, 0.0
            return self.velocity, self.steering

    def setVel(self, move):
        if self.dist2next > 0.7:
            self.velocity = self.max_velocity
        elif self.dist2next > 0.4:
            self.velocity = 30.0
        else:
            self.velocity = 15.0

        if move == -1:
            self.velocity = -self.velocity

    def setSteer(self, yaw, heading):
        if abs(yaw - heading) > 0.03:
            step = 1.0
            adjustment = self.pid.adjust(yaw, heading, step)
            if adjustment >= 0.0:
                self.steering = min(adjustment, 1.0)
            else:
                self.steering = max(adjustment, -1.0)

def main():
    """
    The function to initialize the simulation and return the obtained dataset.
    """
    t0 = time()

    sim = Simulation()

    while True:
        image, dataset, status = sim.collectData(True)

        # Display the image
        if image is not None:
            plt.clf()
            clear_output(wait=True)
            plt.imshow(image)

        if status == -1:
            print('Total run time:', floor((time()-t0)/60), 'min',
                  round((time()-t0)%60, 1), 'sec.')
            plt.show()
            break

        #### For running on Colab
        plt.show(block=True)
        #### For running on local computer
        # plt.show(block=False)
        # plt.pause(0.01)

if __name__ == '__main__':
    main()