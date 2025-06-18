import gymnasium as gym
import highway_env
from matplotlib import pyplot as plt

# 환경 선택
env = gym.make("custom-mixed-road-v0", render_mode="rgb_array") 
obs, info = env.reset()
done = False
truncated = False

# 디버깅 루프
step = 0
while not (done or truncated):
    action = env.unwrapped.action_type.actions_indexes["IDLE"]
    obs, reward, done, truncated, info = env.step(action)

    # 핵심 디버깅 정보 출력
    ego = env.unwrapped.vehicle
    print(
        f"[Step {step}] done: {done}, crashed: {ego.crashed}, pos: {ego.position}, "
        f"speed: {ego.speed:.2f}, reward: {reward:.2f}"
    )
    step += 1

    env.render()

# 마지막 화면 출력
plt.imshow(env.render())
plt.axis("off")
plt.show()