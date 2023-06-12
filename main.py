import numpy as np
import gymnasium as gym
import random,time

class QAgent:
    def __init__(self, env, discount_factor=0.99, learning_rate=0.1, exploration_rate=1.0, exploration_decay_rate=0.001):
        self.env = env

        self.q_table = np.zeros(env.observation_space[2].shape + (env.action_space.n,))

        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.exploration_rate_max = 1.0
        self.exploration_rate_min = 0.01
        self.exploration_decay_rate = exploration_decay_rate

    def get_action(self, state):
        if random.random() < self.exploration_rate:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state, done):
        if done:
            self.q_table[state][action] = reward
        else:
            self.q_table[state][action] = ((1 - self.learning_rate) * self.q_table[state][action] +
                                           self.learning_rate * (reward + self.discount_factor * np.max(
                        self.q_table[next_state])))

    def train(self, max_episodes=10000, max_steps=100):
        rewards = []
        for episode in range(max_episodes):
            state = self.env.reset().get('agent')
            total_reward = 0

            for step in range(max_steps):
                action = self.get_action(state)

                next_state, reward, done, _ = self.env.step(action)

                self.update_q_table(state, action, reward, next_state.get('agent'), done)

                if (episode + 1) % 50 == 0:
                    self.env.render()

                state = next_state.get('agent')
                total_reward += reward

                if done:
                    break
            rewards.append(total_reward)

            self.exploration_rate = (self.exploration_rate_min +
                                     (self.exploration_rate_max - self.exploration_rate_min) *
                                     np.exp(-self.exploration_decay_rate * episode))

            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(rewards[-100:])
                print(
                    f"Episode: {episode + 1}, Average Reward: {avg_reward}, Exploration Rate: {self.exploration_rate}")

            if self.exploration_rate < 0.1:
                break

        return rewards

    def test(self, max_steps=100):

        state = self.env.reset().get('agent')
        total_reward = 0

        for step in range(max_steps):
            action = np.argmax(self.q_table[state])
            next_state, reward, done, _ = self.env.step(action)

            state = next_state.get('agent')
            total_reward += reward

            self.env.render()
            time.sleep(0.1)

            if done:
                print("Total reward:", total_reward)
                break



env = gym.make('gym_maze:Maze-v0', size=15, seed="alamakota")
agent = QAgent(env)
agent.train()
agent.test()
