"""
File including common classes and functions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches



class Scene:

    def __init__(self, start, goal, region, obstacles):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.region = region
        self.obstacles = obstacles

    def plot_scene(self):
        ax = plt.gca()
        for r in self.obstacles:
            rect = patches.Rectangle(r.v1, r.w, r.h, r.angle, color='cyan')
            ax.add_artist(rect)
        plt.axis([self.region.v1[0]-0.5, self.region.v4[0]+0.5, self.region.v1[1]-0.5, self.region.v4[1]+0.5])
        plt.plot(self.start[0], self.start[1], "xr", markersize=10)
        plt.plot(self.goal[0], self.goal[1], "xb", markersize=10)
        plt.legend(('start', 'goal'), loc='upper left')
        plt.gca().set_aspect('equal')

    def show_scene(self):
        plt.figure()
        plot_scene()
        plt.tight_layout()
        plt.show()


