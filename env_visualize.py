import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt

env = gym.make("custom-mixed-road-v0")
env.reset()

road = env.unwrapped.road
fig, ax = plt.subplots(figsize=(12, 10))

xs_all, ys_all = [], []  # 전체 좌표 저장용

# 도로의 모든 차선을 순회하면서 그리기
for lane_id, lane in road.network.lanes_dict().items():
    try:
        points = [lane.position(s, 0) for s in range(0, int(lane.length), 2)]
        xs, ys = zip(*points)
        xs_all.extend(xs)
        ys_all.extend(ys)
        label = f"{lane_id[0]}→{lane_id[1]} ({lane_id[2]})"
        ax.plot(xs, ys, label=label)
    except Exception as e:
        print(f"❌ {lane_id} 시각화 오류: {e}")

# 전체 범위 계산
x_min, x_max = min(xs_all), max(xs_all)
y_min, y_max = min(ys_all), max(ys_all)

# 시각화 설정
ax.set_aspect("equal")
ax.set_title("Mixed_Road Network Visualization")
ax.legend(fontsize=6, loc='upper left', bbox_to_anchor=(1.05, 1))
plt.xlim(x_min - 50, x_max + 50)
plt.ylim(y_min - 50, y_max + 50)
plt.tight_layout()
plt.show()