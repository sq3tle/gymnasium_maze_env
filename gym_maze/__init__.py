from gymnasium.envs.registration import register

register(
    id='Maze-v0',
    entry_point='gym_maze.envs:MazeEnv',
)