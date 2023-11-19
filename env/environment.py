import cv2
import numpy as np

from env.agentKinematics import RoboticAssistant
from utils.utils import SparseDepth
from utils.vision import Vision


class IndoorDeepRL:
    def __init__(self, map_path="complex.png"):
        self.terra = cv2.flip(cv2.imread(map_path), 0)
        self.terra[self.terra > 128] = 255
        self.terra[self.terra <= 128] = 0
        self.m = np.asarray(self.terra)
        self.m = cv2.cvtColor(self.m, cv2.COLOR_RGB2GRAY)
        self.m = self.m.astype(float) / 255.
        self.terra = self.terra.astype(float) / 255.
        self.lmodel = Vision(self.m)

    def createInstance(self):
        self.robot = RoboticAssistant(d=5, wu=9, wv=4, car_w=9, car_f=7, car_r=10, dt=0.1)
        self.robot.x, self.robot.y = self.random_start_travesable()
        self.robot.theta = 360 * np.random.random()
        self.pos = (self.robot.x, self.robot.y, self.robot.theta)

        self.target = self.random_start_travesable()
        self.target_euclidian = np.sqrt((self.robot.x - self.target[0]) ** 2 + (self.robot.y - self.target[1]) ** 2)
        target_angle = np.arctan2(self.target[1] - self.robot.y, self.target[0] - self.robot.x) - np.deg2rad(
            self.robot.theta)
        target_distance = [self.target_euclidian * np.cos(target_angle), self.target_euclidian * np.sin(target_angle)]

        self.sdata, self.plist = self.lmodel.measure_depth(self.pos)
        state = self.existance(self.sdata, target_distance)
        return state

    def step(self, action):
        self.robot.control((action[0] + 1) / 2 * self.robot.v_interval, action[1] * self.robot.w_interval)
        self.robot.update()

        e1, e2, e3, e4 = self.robot.dimensions
        ee1 = SparseDepth(e1[0], e2[0], e1[1], e2[1])
        ee2 = SparseDepth(e1[0], e3[0], e1[1], e3[1])
        ee3 = SparseDepth(e3[0], e4[0], e3[1], e4[1])
        ee4 = SparseDepth(e4[0], e2[0], e4[1], e2[1])
        check = ee1 + ee2 + ee3 + ee4

        collision = False
        for points in check:
            if self.m[int(points[1]), int(points[0])] < 0.5:
                collision = True
                self.robot.redo()
                self.robot.velocity = -0.5 * self.robot.velocity
                break

        self.pos = (self.robot.x, self.robot.y, self.robot.theta)
        self.sdata, self.plist = self.lmodel.measure_depth(self.pos)

        action_r = 0.05 if action[0] < -0.5 else 0

        curr_target_dist = np.sqrt((self.robot.x - self.target[0]) ** 2 + (self.robot.y - self.target[1]) ** 2)
        distance_reward = self.target_euclidian - curr_target_dist

        s_orien = np.rad2deg(np.arctan2(self.target[1] - self.robot.y, self.target[0] - self.robot.x))
        orientation_error = (s_orien - self.robot.theta) % 360
        if orientation_error > 180:
            orientation_error = 360 - orientation_error
        orientation_reward = np.deg2rad(orientation_error)

        reward = distance_reward - orientation_reward - 0.6 * action_r

        terminated = False

        if curr_target_dist < 20:
            reward = 20
            terminated = True
        if collision:
            reward = -15
            terminated = True

        self.target_euclidian = curr_target_dist
        target_angle = np.arctan2(self.target[1] - self.robot.y, self.target[0] - self.robot.x) - np.deg2rad(
            self.robot.theta)
        target_distance = [self.target_euclidian * np.cos(target_angle), self.target_euclidian * np.sin(target_angle)]
        state_next = self.existance(self.sdata, target_distance)

        return state_next, reward, terminated

    def render(self, gui=True):
        experiment_space = self.terra.copy()
        for pts in self.plist:
            cv2.line(
                experiment_space,
                (int(1 * self.pos[0]), int(1 * self.pos[1])),
                (int(1 * pts[0]), int(1 * pts[1])),
                (0.0, 1.0, 0.0), 1)

        cv2.circle(experiment_space, (int(1 * self.target[0]), int(1 * self.target[1])), 10, (1.0, 0.5, 0.7), 3)
        experiment_space = self.robot.render(experiment_space)
        experiment_space = cv2.flip(experiment_space, 0)
        if gui:
            cv2.imshow("Mapless Navigation", experiment_space)
            k = cv2.waitKey(1)

        return experiment_space.copy()

    def random_start_travesable(self):
        height, width = self.m.shape[0], self.m.shape[1]
        tx = np.random.randint(0, width)
        ty = np.random.randint(0, height)

        kernel = np.ones((10, 10), np.uint8)
        m_dilate = 1 - cv2.dilate(1 - self.m, kernel, iterations=3)
        while (m_dilate[ty, tx] < 0.5):
            tx = np.random.randint(0, width)
            ty = np.random.randint(0, height)
        return tx, ty

    def existance(self, sensor, target):
        si = [s / 200 for s in sensor]
        ti = [t / 500 for t in target]
        return si + ti
