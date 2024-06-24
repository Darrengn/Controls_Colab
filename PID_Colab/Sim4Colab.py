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

        self.sim.create_env(
            env_config=ENV_PATH,
            GUI=False,
            planned_path=planned_path
        )

        # Set simulation response time
        self.path_sim_time = Timings(PATH_SIM_FPS)
        self.lidar_time = Timings(LIDAR_FPS)
        self.print_frequency = Timings(PRINT_FPS)
        self.collect_data_time = Timings(COLLECT_DATA_FPS)
        self.capture_image_time = Timings(IMAGE_FPS)

        # Initialize the car
        self.car = Car(
            urdf_path=CAR_URDF_PATH,
            car_config=CAR_PATH,
            sensors=SENSORS
        )

        self.car.place_car(
            self.sim.floor,
            xy_coord=(planned_path[0][0], planned_path[0][1])
        )

        # Initialize path simulator
        self.path_sim = PathSimulator(self.car, PATH_SIM_FPS, planned_path)

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
                print('Arrived at destination.')
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
            path,
        ):

        self.car = car
        self.sim_fps = sim_fps
        self.max_velocity = 60.0  # Maximum velocity of 60 units per second

        self.velocity = 0
        self.steering = 0
        self.pid = PID()

        # Load path from file
        self.path = path
        self.next = 0  # Start from the first waypoint
        self.length = len(self.path)
        self.dist2next = 0
        self.ifTurn = False

    def navigate(self, x, y, yaw):
        if self.next >= len(self.path):
            return float('inf'), float('inf')

        next_x, next_y = self.path[self.next]
        heading = atan2(next_y - y, next_x - x)
        move = 1  # Always move forward

        self.dist2next = np.linalg.norm(
            np.array((next_x, next_y)) - np.array((x, y)))

        # Turn
        if self.ifTurn == False:
            if abs(heading - yaw) <= 0.15:
                self.ifTurn = True
                self.steering = 0.0
                return 0.0, 0.0

            adjustment = self.pid.adjust(yaw, heading, 1.0 / self.sim_fps)
            self.steering = max(min(adjustment, 1.0), -1.0)

            return 15.0, self.steering

        # Move
        if self.dist2next >= 0.2:
            self.setVel(move)
            self.setSteer(yaw, heading)
            return self.velocity, self.steering
        else:
            self.next += 1
            print('Moving to next waypoint [', self.next, '/', self.length, '].')
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
        adjustment = self.pid.adjust(yaw, heading, 1.0 / self.sim_fps)
        self.steering = max(min(adjustment, 1.0), -1.0)

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