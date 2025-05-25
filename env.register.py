import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt


env = gym.make("custom-mixed-road-v0")
env.reset()

road = env.unwrapped.road
fig, ax = plt.subplots(figsize=(12, 10))

# 도로의 모든 차선을 순회하면서 그리기
for lane_id, lane in road.network.lanes_dict().items():
    try:
        points = [lane.position(s, 0) for s in range(0, int(lane.length), 2)]
        xs, ys = zip(*points)
        label = f"{lane_id[0]}→{lane_id[1]} ({lane_id[2]})"
        ax.plot(xs, ys, label=label)
    except Exception as e:
        print(f"❌ {lane_id} 시각화 오류: {e}")

ax.set_aspect("equal")
ax.set_title("1차 Mixed_Road Network Visualization")
ax.legend(fontsize=6, loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()