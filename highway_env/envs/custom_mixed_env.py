# highway_env/envs/custom_mixed_env.py

import numpy as np

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import CircularLane, LineType, SineLane, StraightLane
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.objects import Obstacle
from highway_env import utils
from highway_env.vehicle.controller import ControlledVehicle

class MixedRoadEnv(AbstractEnv):
    @classmethod
    def default_config(self):
        config = super().default_config()
        config.update(
            {
                # 기본 전역 설정(from HighwayEnv)
                "observation": {"type": "Kinematics"},
                "action": {"type": "DiscreteMetaAction"},
                "lanes_count": 4,
                "vehicles_count": 50,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 40,
                "ego_spacing": 2,
                "vehicles_density": 1,
                "normalize_reward": True,
                "offroad_terminal": False,

                # segment별 보상 정책
                "segment_configs": {
                    "merge": {
                        "collision_reward": -1,
                        "right_lane_reward": 0.1,
                        "high_speed_reward": 0.2,
                        "reward_speed_range": [20, 30],
                        "merging_speed_reward": -0.5,
                        "lane_change_reward": -0.05,
                    },
                    "highway": {
                        "collision_reward": -1,
                        "right_lane_reward": 0.1,
                        "high_speed_reward": 0.4,
                        "lane_change_reward": 0,
                        "on_road_reward": 0.1,
                        "reward_speed_range": [20, 30],
                        "normalize_reward": True,
                    },
                },
            }
    )
        return config

    def _make_road(self):
        """고속도로 -> 2차선 교차로 -> 회전 교차로 순서의 환경을 생성합니다. 
        고속도로는 직선 도로로 구성되며, 2차선 교차로는 두 개의 직선 도로가 연결됩니다.
        회전 교차로는 원형 도로로 구성됩니다. 원형 교차로를 빠져나와 직선 구선에서 주행을 
        완료하는 환경을 생서합니다.
        """
        net = RoadNetwork()

        # Highway lanes
        ends = [150, 80, 80, 150]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]

        for i in range(2):
            net.add_lane(
                "hw_a",
                "hw_b",
                StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]),
            )
            net.add_lane(
                "hw_b",
                "hw_c",
                StraightLane(
                    [sum(ends[:2]), y[i]],
                    [sum(ends[:3]), y[i]],
                    line_types=line_type_merge[i],
                ),
            )
            net.add_lane(
                "hw_c",
                "ex_a",
                StraightLane(
                    [sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i]
                ),
            )

        # Merging lanes
        amplitude = 3.25
        ljk = StraightLane(
            [0, 6.5 + 4 + 4], [ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True
        )
        lkb = SineLane(
            ljk.position(ends[0], -amplitude),
            ljk.position(sum(ends[:2]), -amplitude),
            amplitude,
            2 * np.pi / (2 * ends[1]),
            np.pi / 2,
            line_types=[c, c],
            forbidden=True,
        )
        # lbc = StraightLane(
        #     lkb.position(ends[1], 0),
        #     lkb.position(ends[1], 0) + [ends[2], 0],
        #     line_types=[n, c],
        #     forbidden=True,
        # )
        net.add_lane("mg_j", "mg_k", ljk)
        net.add_lane("mg_k", "hw_b", lkb)
       # net.add_lane("b", "c", lbc)
      
        merge_lane = net.get_lane(("hw_b", "hw_c", 0))
        merge_end = merge_lane.position(merge_lane.length, 0)

        # net.add_lane(
        #     "c", "ser",
        #     StraightLane(merge_end, [2, 170], line_types=[LineType.CONTINUOUS, LineType.CONTINUOUS])
        # )
        
        # # 회전 교차로 시작  
        # center = [merge_end[0] + 40, merge_end[1]] # [m]
        # radius = 20  # [m]
        # alpha = 24  # [deg]

        # radii = [radius, radius + 4]
        # line = [[c, s], [n, c]]
        # for lane in [0, 1]:
        #     net.add_lane(
        #         "se",
        #         "ex",
        #         CircularLane(
        #             center,
        #             radii[lane],
        #             np.deg2rad(90 - alpha),
        #             np.deg2rad(alpha),
        #             clockwise=False,
        #             line_types=line[lane],
        #         ),
        #     )
        #     net.add_lane(
        #         "ex",
        #         "ee",
        #         CircularLane(
        #             center,
        #             radii[lane],
        #             np.deg2rad(alpha),
        #             np.deg2rad(-alpha),
        #             clockwise=False,
        #             line_types=line[lane],
        #         ),
        #     )
        #     net.add_lane(
        #         "ee",
        #         "nx",
        #         CircularLane(
        #             center,
        #             radii[lane],
        #             np.deg2rad(-alpha),
        #             np.deg2rad(-90 + alpha),
        #             clockwise=False,
        #             line_types=line[lane],
        #         ),
        #     )
        #     net.add_lane(
        #         "nx",
        #         "ne",
        #         CircularLane(
        #             center,
        #             radii[lane],
        #             np.deg2rad(-90 + alpha),
        #             np.deg2rad(-90 - alpha),
        #             clockwise=False,
        #             line_types=line[lane],
        #         ),
        #     )
        #     net.add_lane(
        #         "ne",
        #         "wx",
        #         CircularLane(
        #             center,
        #             radii[lane],
        #             np.deg2rad(-90 - alpha),
        #             np.deg2rad(-180 + alpha),
        #             clockwise=False,
        #             line_types=line[lane],
        #         ),
        #     )
        #     net.add_lane(
        #         "wx",
        #         "we",
        #         CircularLane(
        #             center,
        #             radii[lane],
        #             np.deg2rad(-180 + alpha),
        #             np.deg2rad(-180 - alpha),
        #             clockwise=False,
        #             line_types=line[lane],
        #         ),
        #     )
        #     net.add_lane(
        #         "we",
        #         "sx",
        #         CircularLane(
        #             center,
        #             radii[lane],
        #             np.deg2rad(180 - alpha),
        #             np.deg2rad(90 + alpha),
        #             clockwise=False,
        #             line_types=line[lane],
        #         ),
        #     )
        #     net.add_lane(
        #         "sx",
        #         "se",
        #         CircularLane(
        #             center,
        #             radii[lane],
        #             np.deg2rad(90 + alpha),
        #             np.deg2rad(90 - alpha),
        #             clockwise=False,
        #             line_types=line[lane],
        #         ),
        #     )

        # # Access lanes: (r)oad/(s)ine
        # access = 170  # [m]
        # dev = 85  # [m]
        # a = 5  # [m]
        # delta_st = 0.2 * dev  # [m]

        # delta_en = dev - delta_st
        # w = 2 * np.pi / dev
        # net.add_lane(
        #     "ser", "ses", StraightLane([2, access], [2, dev / 2], line_types=(s, c))
        # )
        # net.add_lane(
        #     "ses",
        #     "se",
        #     SineLane(
        #         [2 + a, dev / 2],
        #         [2 + a, dev / 2 - delta_st],
        #         a,
        #         w,
        #         -np.pi / 2,
        #         line_types=(c, c),
        #     ),
        # )
        # net.add_lane(
        #     "sx",
        #     "sxs",
        #     SineLane(
        #         [-2 - a, -dev / 2 + delta_en],
        #         [-2 - a, dev / 2],
        #         a,
        #         w,
        #         -np.pi / 2 + w * delta_en,
        #         line_types=(c, c),
        #     ),
        # )
        # net.add_lane(
        #     "sxs", "sxr", StraightLane([-2, dev / 2], [-2, access], line_types=(n, c))
        # )

        # net.add_lane(
        #     "eer", "ees", StraightLane([access, -2], [dev / 2, -2], line_types=(s, c))
        # )
        # net.add_lane(
        #     "ees",
        #     "ee",
        #     SineLane(
        #         [dev / 2, -2 - a],
        #         [dev / 2 - delta_st, -2 - a],
        #         a,
        #         w,
        #         -np.pi / 2,
        #         line_types=(c, c),
        #     ),
        # )
        # net.add_lane(
        #     "ex",
        #     "exs",
        #     SineLane(
        #         [-dev / 2 + delta_en, 2 + a],
        #         [dev / 2, 2 + a],
        #         a,
        #         w,
        #         -np.pi / 2 + w * delta_en,
        #         line_types=(c, c),
        #     ),
        # )
        # net.add_lane(
        #     "exs", "exr", StraightLane([dev / 2, 2], [access, 2], line_types=(n, c))
        # )

        # net.add_lane(
        #     "ner", "nes", StraightLane([-2, -access], [-2, -dev / 2], line_types=(s, c))
        # )
        # net.add_lane(
        #     "nes",
        #     "ne",
        #     SineLane(
        #         [-2 - a, -dev / 2],
        #         [-2 - a, -dev / 2 + delta_st],
        #         a,
        #         w,
        #         -np.pi / 2,
        #         line_types=(c, c),
        #     ),
        # )
        # net.add_lane(
        #     "nx",
        #     "nxs",
        #     SineLane(
        #         [2 + a, dev / 2 - delta_en],
        #         [2 + a, -dev / 2],
        #         a,
        #         w,
        #         -np.pi / 2 + w * delta_en,
        #         line_types=(c, c),
        #     ),
        # )
        # net.add_lane(
        #     "nxs", "nxr", StraightLane([2, -dev / 2], [2, -access], line_types=(n, c))
        # )

        # net.add_lane(
        #     "wer", "wes", StraightLane([-access, 2], [-dev / 2, 2], line_types=(s, c))
        # )
        # net.add_lane(
        #     "wes",
        #     "we",
        #     SineLane(
        #         [-dev / 2, 2 + a],
        #         [-dev / 2 + delta_st, 2 + a],
        #         a,
        #         w,
        #         -np.pi / 2,
        #         line_types=(c, c),
        #     ),
        # )
        # net.add_lane(
        #     "wx",
        #     "wxs",
        #     SineLane(
        #         [dev / 2 - delta_en, -2 - a],
        #         [-dev / 2, -2 - a],
        #         a,
        #         w,
        #         -np.pi / 2 + w * delta_en,
        #         line_types=(c, c),
        #     ),
        # )
        # net.add_lane(
        #     "wxs", "wxr", StraightLane([-dev / 2, -2], [-access, -2], line_types=(n, c))
        # )
    
        self.segment_labels = {
        ("hw_a", "hw_b"): "highway_1",
        ("hw_b", "hw_c"): "highway_2",
        ("hw_c", "ex_a"): "highway_exit",
        ("mg_j", "mg_k"): "merge_straight",
        ("mg_k", "hw_b"): "merge_entry"
     }   

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road


    def _make_vehicles(self):
        # 주행 에이전트 
        vehicle = self.action_type.vehicle_class(
            self.road, self.road.network.get_lane(("hw_a", "b", 0)).position(5, 0)
        )
        self.vehicle = vehicle
        self.road.vehicles.append(vehicle)

        # 다른 차량들 생성
        for _ in range(self.config["vehicles_count"]):
            v = IDMVehicle.create_random(self.road)
            self.road.vehicles.append(v)

    def _reset(self):
        self._make_road()
        self._make_vehicles()

    def _reward(self, action) -> float:
        rewards = self._rewards(action)

        lane_index = self.vehicle.lane_index
        if not lane_index:
            return 0.0

        _from, _to, _ = lane_index
        segment_key = (_from, _to)
        segment_type = self.segment_labels.get(segment_key, "default")
        segment_config = self.config["segment_configs"].get(segment_type, {})

        # 보상 가중합 계산
        reward = sum(
            segment_config.get(name, 0) * value
            for name, value in rewards.items()
        )

        # normalize
        if self.config.get("normalize_reward", False):
            min_reward = 0
            max_reward = 0

            if segment_type.startswith("merge"):
                min_reward = (
                    segment_config.get("collision_reward", 0)
                    + segment_config.get("merging_speed_reward", 0)
                )
                max_reward = (
                    segment_config.get("high_speed_reward", 0)
                    + segment_config.get("right_lane_reward", 0)
                )
            elif segment_type.startswith("highway"):
                min_reward = segment_config.get("collision_reward", 0)
                max_reward = (
                    segment_config.get("high_speed_reward", 0)
                    + segment_config.get("right_lane_reward", 0)
                )

            reward = utils.lmap(reward, [min_reward, max_reward], [0, 1])

        # highway에서만 on_road_reward 곱하기
        if segment_type.startswith("highway") and "on_road_reward" in rewards:
            reward *= rewards["on_road_reward"]

        return reward
   
    
    def _rewards(self, action) -> dict[str, float]:
    # 현재 차량 위치 기반 세그먼트 타입 가져오기
        lane_index = self.vehicle.lane_index
        if not lane_index:
            return {}

        _from, _to, _ = lane_index
        segment_key = (_from, _to)
        segment_type = self.segment_labels.get(segment_key, "default")

        # 해당 세그먼트의 보상 설정 가져오기
        segment_config = self.config["segment_configs"].get(segment_type, {})

        # 보상 요소 계산
        rewards = {}

        # 공통 충돌 보상 
        if "collision_reward" in segment_config:
            rewards["collision_reward"] = float(self.vehicle.crashed)
        
        ## highway 구간
        # 고속 보상 (forward_speed 기준)
        if segment_type.startswith("highway") and  "high_speed_reward" in segment_config:
            forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
            scaled_speed = utils.lmap(
                forward_speed, self.config["reward_speed_range"], [0, 1])
            
            rewards["high_speed_reward"] = np.clip(scaled_speed, 0, 1)

        if segment_type.startswith("highway") and "right_lane_reward" in segment_config:
            neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
            lane = (
                self.vehicle.target_lane_index[2]
                if isinstance(self.vehicle, ControlledVehicle)
                else self.vehicle.lane_index[2])
            rewards["right_lane_reward"] = lane / max(len(neighbours) - 1, 1)

        # one road 보상
        if segment_type.startswith("highway") and "on_road_reward" in segment_config:
            rewards["on_road_reward"] = float(self.vehicle.on_road)


        ## merge 구간         
        if segment_type.startswith("merge") and "right_lane_reward" in segment_config:
            rewards["right_lane_reward"] = self.vehicle.lane_index[2] / 1
        
        if segment_type.startswith("merge") and "high_speed_reward" in segment_config:
            scaled_speed = utils.lmap(
            self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
            rewards["high_speed_reward"] = scaled_speed

        # 고속 보상
        if segment_type.startswith("merge") and "lane_change_reward" in segment_config:
            rewards["lane_change_reward"] = float(action in [0, 2])

        # 차선 변경 보
        if segment_type.startswith("merge") and "merging_speed_reward" in segment_config:
            rewards["merging_speed_reward"] = sum(  # Altruistic penalty
                (vehicle.target_speed - vehicle.speed) / vehicle.target_speed
                for vehicle in self.road.vehicles
                if self.segment_labels.get(vehicle.lane_index[:2], "").startswith("merge")
                and isinstance(vehicle, ControlledVehicle)
            )
    
        return rewards
    
    def _is_terminated(self) -> bool:
        return (
            any(v.crashed for v in self.controlled_vehicles)
            or all(self.has_arrived(v) for v in self.controlled_vehicles)
        )
    
    def has_arrived(self, vehicle, exit_distance=25) -> bool:
        if not vehicle.lane_index:
            return False
        _from, _to, _ = vehicle.lane_index
        return (
            _from == "c" and _to == "d"
            and vehicle.lane.local_coordinates(vehicle.position)[0] >= vehicle.lane.length - exit_distance
        )
    
    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]
    
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "simulation_frequency": 5,
                "lanes_count": 3,
                "vehicles_count": 20,
                "duration": 30,  # [s]
                "ego_spacing": 1.5,
            }
        )
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False