import gymnasium as gym
import highway_env
from matplotlib import pyplot as plt


# env = gym.make("custom-mixed-road-v0", render_mode="rgb_array")
env = gym.make("merge-v0", render_mode="rgb_array")

env.reset()

road = env.unwrapped.road
for _ in range(10):
    action = env.unwrapped.action_type.actions_indexes["IDLE"]
    obs, reward, done, truncated, info = env.step(action)
    env.render()

plt.imshow(env.render())
plt.show()