import random

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces



class MazeEnv(gym.Env):
    def __init__(self, seed=None, size=None):
        super(MazeEnv, self).__init__()

        self.size = size if size else 8
        self.maze = self._gen_maze(seed, size)

        pygame.init()
        self.screen = None
        self.cell_size = int(800/self.size)
        self.width = self.maze.shape[1] * self.cell_size
        self.height = self.maze.shape[0] * self.cell_size

        self.start = tuple(np.argwhere(self.maze == 2)[0])
        self.end = tuple(np.argwhere(self.maze == 3)[0])
        self.current_position = self.start

        self.n_actions = 4

        self.total_steps = 0

        self.action_space = spaces.Discrete(self.n_actions)

        self.observation_space = spaces.Tuple((spaces.Discrete(self.maze.shape[0]),
                                               spaces.Discrete(self.maze.shape[1]),
                                               spaces.Box(low=0, high=2,
                                                          shape=self.maze.shape,
                                                          dtype=np.uint8)))

    def step(self, action):
        self.total_steps += 1

        if action == 0:  # Up
            next_position = (self.current_position[0] - 1, self.current_position[1])
        elif action == 1:  # Right
            next_position = (self.current_position[0], self.current_position[1] + 1)
        elif action == 2:  # Down
            next_position = (self.current_position[0] + 1, self.current_position[1])
        elif action == 3:  # Left
            next_position = (self.current_position[0], self.current_position[1] - 1)
        else:
            raise ValueError("Invalid action")

        if (next_position[0] < 0 or next_position[0] >= self.maze.shape[0] or
                next_position[1] < 0 or next_position[1] >= self.maze.shape[1] or
                self.maze[next_position] == 1):
            next_position = self.current_position

        self.current_position = next_position

        if self.current_position == self.end:
            reward = 100
            done = True
        else:
            reward = -1
            done = False

        info = {}

        observation = self._get_obs()

        return observation, reward, done, info

    def reset(self, seed=None, options=None):
        self.current_position = self.start
        return self._get_obs()

    def render(self, mode='human'):
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.width + 2 * self.cell_size, self.height + 2 * self.cell_size))

        maze_copy = self.maze.copy()
        maze_copy[self.current_position] = 4
        pygame.time.Clock().tick(144)
        self._gui(maze_copy)

    def _get_obs(self):
        return {"agent": self.current_position, "target": self.end, "maze": self.maze}

    def _gen_maze(self, seed=None, size=8):

        if seed:
            random.seed(seed)

        maze = np.ones((size, size), dtype=np.int8)
        start = (0, 1)
        exit = (size - 1, size - 2)
        stack = [start]

        DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        while stack:
            x, y = stack[-1]
            maze[x, y] = 0
            neighbors = [(x + dx, y + dy) for dx, dy in DIRS if 0 <= x + dx < size and 0 <= y + dy < size]

            neighbors = [(nx, ny) for nx, ny in neighbors if maze[nx, ny] == 1 and
                         len([(nx + dx, ny + dy) for dx, dy in DIRS if 0 <= nx + dx < size and
                              0 <= ny + dy < size and
                              maze[nx + dx, ny + dy] == 0]) <= 1]

            if neighbors:
                next_x, next_y = neighbors[random.randint(0, len(neighbors) - 1)]

                maze[next_x, next_y] = 0
                stack.append((next_x, next_y))
            else:
                stack.pop()

        maze[start] = 2
        maze[exit] = 3

        return maze

    def _gui(self, maze):
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        TAKI_ZGASZONY_CYAN = (0, 200, 200)
        ORANGE = (255, 170, 0)
        RED = (255, 0, 0)

        self.screen.fill(BLACK)

        pygame.draw.rect(self.screen, WHITE, (self.cell_size, self.cell_size, self.width, self.height))

        for row in range(len(maze)):
            for col in range(len(maze[0])):
                cell = maze[row][col]
                x = (col + 1) * self.cell_size
                y = (row + 1) * self.cell_size

                if cell == 0:
                    pygame.draw.rect(self.screen, WHITE, (x, y, self.cell_size, self.cell_size))
                elif cell == 1:
                    pygame.draw.rect(self.screen, BLACK, (x, y, self.cell_size, self.cell_size))
                elif cell == 2:
                    pygame.draw.rect(self.screen, TAKI_ZGASZONY_CYAN, (x, y, self.cell_size, self.cell_size))
                elif cell == 3:
                    pygame.draw.rect(self.screen, ORANGE, (x, y, self.cell_size, self.cell_size))
                elif cell == 4:
                    pygame.draw.rect(self.screen, RED, (x, y, self.cell_size, self.cell_size))

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
