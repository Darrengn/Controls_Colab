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

import numpy as np
from math import *
import matplotlib.pyplot as plt
from IPython.display import clear_output
from time import time
from util import World2Grid, Grid2World

class PID:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0.0
        self.integral = 0.0

    def control(self, error, delta_time):
        self.integral += error * delta_time
        derivative = (error - self.previous_error) / delta_time
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


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

        # planned_path = [
        #     [0.0, -1.95], [-1.95, -1.95], [-1.95, 1.95], [1.95, 1.95],
        #     [1.95, -1.95], [0.0, -1.95], [0.0, 1.95], [-1.95, 1.95],
        #     [-1.95, -1.95], [1.95, -1.95], [1.95, 1.95], [0.0, 1.95],
        #     [0.0, 0.0]
        # ]

        # map_size = 5.0
        # res = 0.01
        # grid_size = int(map_size/res)
        # planned_path = [World2Grid(pt, map_size, grid_size) for pt in planned_path]

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

        # Initial the car
        self.car = Car(
            urdf_path=CAR_URDF_PATH,
            car_config=CAR_PATH,
            sensors=SENSORS
        )

        starting_point = Grid2World((planned_path[0][0], 500 - planned_path[0][1]), 5.0, 500, 0.01)

        print('Starting point:', starting_point)

        self.car.place_car(
            self.sim.floor,
            xy_coord=starting_point,
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

class PathSimulator:
    def __init__(self, car, sim_fps):
        self.car = car
        self.sim_fps = sim_fps
        self.max_velocity = 60.0  # Maximum velocity of 60 units per second

        self.velocity = 0
        self.steering = 0
        self.pid = PID(kp=1.0, ki=0.1, kd=0.05)

        self.planned_path = np.load('/content/Controls_Colab/PID_Colab/trajectory.npy')
        self.planned_path = [(x, y) for x, y in self.planned_path]

        self.waypoints = {}
        for i, (x, y) in enumerate(self.planned_path):
            x_real, y_real = Grid2World((x, 500-y), 5.0, 500, 0.01)
            self.waypoints[i+1] = (x_real, y_real)

        print('Waypoints:', self.waypoints)

        self.next = 1  # waypoint number
        self.length = len(self.waypoints)
        self.dist2next = 0
        self.ifTurn = False
        self.max_time_per_waypoint = 5.0  # maximum time to reach a waypoint (seconds)
        self.time_spent = 0.0  # time spent at current waypoint
        self.lookahead_distance = 5.0  # lookahead distance in world units

    def find_lookahead_point(self, x, y):
        lookahead_point = None
        for i in range(self.next, self.length + 1):
            waypoint = self.waypoints[i]
            waypoint_x, waypoint_y = waypoint
            distance = np.hypot(waypoint_x - x, waypoint_y - y)
            if distance > self.lookahead_distance:
                lookahead_point = waypoint
                self.next = i
                break
        if lookahead_point is None:
            lookahead_point = self.waypoints[self.length]
        return lookahead_point

    def navigate(self, x, y, yaw):
        if self.next > self.length:
            return 0, 0  # No more waypoints to follow

        lookahead_point = self.find_lookahead_point(x, y)
        waypoint_x, waypoint_y = lookahead_point

        # Calculate cross-track error
        dx = waypoint_x - x
        dy = waypoint_y - y
        cross_track_error = np.hypot(dx, dy)

        # Calculate heading to the lookahead point
        desired_heading = atan2(dy, dx)
        heading_error = desired_heading - yaw
        if heading_error > pi:
            heading_error -= 2 * pi
        elif heading_error < -pi:
            heading_error += 2 * pi

        # Use PID control for steering
        delta_time = 1 / self.sim_fps
        steering = self.pid.control(heading_error, delta_time)
        steering = np.clip(steering, -1, 1)  # Limit steering to [-1, 1]

        # Proportional control for velocity
        velocity = self.max_velocity * min(1, cross_track_error / self.lookahead_distance)
        velocity = np.clip(velocity, -self.max_velocity, self.max_velocity)  # Limit velocity to [-max_velocity, max_velocity]

        # Check if the agent has reached the lookahead point
        if cross_track_error < 1.0:
            self.next += 1
            self.time_spent = 0.0

        # Timeout mechanism to avoid getting stuck
        self.time_spent += delta_time
        if self.time_spent > self.max_time_per_waypoint:
            print(f"Agent stuck at waypoint {self.next}. Skipping to the next waypoint.")
            self.next += 1
            self.time_spent = 0.0

        return velocity, steering

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