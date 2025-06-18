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
                "vehicles_count": 10,  # 차량 수 대폭 줄임
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 40,
                "ego_spacing": 2.5,
                "vehicles_density": 0.1,  # 밀도도 줄임
                "normalize_reward": True,
                "offroad_terminal": False,
                "collision_reward": -1,  # 충돌 보상 명시
                "simulation_frequency": 10,  # 시뮬레이션 주파수 낮춤

                # ================================================================
                # 목표 설정: 회전교차로 탈출 목표
                # ================================================================
                "roundabout_exit_target": "north",  # 목표 출구: north, south, east, west 중 선택
                "success_reward": 10.0,             # 성공적 탈출시 보상
                "completion_distance": 50,          # 출구에서 이 거리만큼 나가면 완료

                "segment_configs": {
                    "default": {
                        "collision_reward": -1,
                        "high_speed_reward": 0.3
                    },
                    
                    # 고속도로 구간: 빠르게 직진, 속도 중심, 차선 이탈 패널티
                    "highway": {
                        "collision_reward": -1,           # 충돌 패널티
                        "high_speed_reward": 0.4,         # 고속 주행 보상 (주요)
                        "right_lane_reward": 0.1,         # 우측 차선 유지 보상
                        "lane_change_reward": 0.2,        # 차선 변경 보상 증가 (적절한 상황에서)
                        "on_road_reward": 0.2,            # 도로 내 유지 보상
                        "off_road_penalty": -0.5,         # 차선 이탈 패널티
                        "reward_speed_range": [25, 35],   # 목표 속도 범위
                        "normalize_reward": True,
                        "overtaking_reward": 0.3,         # 추월 보상 추가
                        "safe_following_reward": 0.1,     # 안전한 거리 유지 보상
                    },
                    
                    # 합류 구간: 우측 진입차량과 병합, 안전거리 유지, 차선 변경 보상
                    "merge": {
                        "collision_reward": -1,           
                        "safe_distance_reward": 0.3,     
                        "merge_cooperation_reward": 0.2,  
                        "lane_change_reward": 0.3,       
                        "speed_adaptation_reward": 0.15,  
                        "right_lane_reward": 0.1,      
                        "high_speed_reward": 0.1,        
                        "reward_speed_range": [20, 30],  
                        "blocking_penalty": -0.3,      
                        "overtaking_reward": 0.25,     
                    },
                    
                    # 회전교차로 구간: 진입 후 목표 출구로 탈출, 차선 유지, 중앙 충돌 회피
                    "roundabout": {
                        "collision_reward": -1,           
                        "lane_keeping_reward": 0.3,      
                        "smooth_turning_reward": 0.25,   
                        "progress_reward": 0.2,         
                        "entry_success_reward": 0.15,   
                        "exit_preparation_reward": 0.1,  
                        "speed_control_reward": 0.1,     
                        "reward_speed_range": [15, 25],  
                        "center_collision_penalty": -0.8, 
                        "wrong_direction_penalty": -0.5,  
                        "target_approach_reward": 0.4,   
                    },
                    
                    # 회전교차로 출구 구간: 성공적 탈출 완료
                    "roundabout_exit": {
                        "collision_reward": -1,
                        "high_speed_reward": 0.3,
                        "completion_reward": 1.0,        # 완료 보상
                    },
                },
            }
    )
        return config

    def _make_road(self):
        """
        전체 구조:
        1. 고속도로 구간 (Highway Section): 2차선 직선 도로 (hw_a -> hw_b -> hw_c)
        2. 합류 구간 (Merge Section): 측면에서 합류하는 차선 (mg_j -> mg_k -> hw_b)  
        3. 회전교차로 구간 (Roundabout Section): 4방향 2차선 원형 교차로
        4. 출구 구간 (Exit Section): 목표 출구를 통한 탈출 경로
        
        각 구간은 서로 다른 주행 특성과 보상 체계를 가집니다.
        """
        net = RoadNetwork()

        # ================================================================
        # 1. 고속도로 구간 (Highway Section)
        # ================================================================
        ends = [150, 80, 80, 150]  # [hw_a->hw_b, hw_b->hw_c(합류구간), hw_c->roundabout연결, roundabout이후]
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]  # 2차선 Y좌표 [0, 4]
        line_type = [[c, s], [n, c]]         # 일반 구간: 중앙선 연속, 차선 점선
        line_type_merge = [[c, s], [n, s]]   # 합류 구간: 양쪽 모두 점선으로 차선변경 허용

        # hw_a -> hw_b: 합류 전 일반 고속도로 구간
        for i in range(2):
            net.add_lane(
                "hw_a",
                "hw_b", 
                StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]),
            )
        
        # hw_b -> hw_c: 합류가 발생하는 구간 (차선 변경이 더 자유로움)
        for i in range(2):
            net.add_lane(
                "hw_b",
                "hw_c",
                StraightLane(
                    [sum(ends[:2]), y[i]],
                    [sum(ends[:3]), y[i]], 
                    line_types=line_type_merge[i],
                ),
            )

        # ================================================================
        # 2. 합류 구간 (Merge Section) 
        # ================================================================
        amplitude = 3.25  # 사인파 진폭 (합류 곡선의 최대 변위)
        
        # mg_j -> mg_k: 합류 전 직선 준비 구간
        
        ljk = StraightLane(
            [0, 6.5 + 4 + 4],           # 시작점: 고속도로 위쪽
            [ends[0], 6.5 + 4 + 4],     # 끝점: hw_a->hw_b 경계까지
            line_types=[c, c], 
            forbidden=True              # 일반 차량 진입 금지 (합류 전용)
        )
        
        lkb = SineLane(
            ljk.position(ends[0], -amplitude),      # 사인파 시작점
            ljk.position(sum(ends[:2]), -amplitude), # 사인파 끝점  
            amplitude,                               # 진폭
            2 * np.pi / (2 * ends[1]),              # 주파수 (합류 구간 길이에 맞춤)
            np.pi / 2,                              # 위상 (아래쪽으로 합류)
            line_types=[c, c],
            forbidden=True
        )
       
        net.add_lane("mg_j", "mg_k", ljk)
        net.add_lane("mg_k", "hw_b", lkb)

        # ================================================================
        # 3. 회전교차로 구간 (Roundabout Section)
        # ================================================================
        # hw_c 차선들의 실제 끝점 좌표 계산 (회전교차로 연결 기준점)
        merge_lane_0 = net.get_lane(("hw_b", "hw_c", 0))
        merge_lane_1 = net.get_lane(("hw_b", "hw_c", 1))
        merge_end_0 = merge_lane_0.position(merge_lane_0.length, 0)  # 상단 차선 끝점
        merge_end_1 = merge_lane_1.position(merge_lane_1.length, 0)  # 하단 차선 끝점

        # 회전교차로 기본 설정 (roundabout_env.py 표준 구조 적용)
        access = 170     
        dev = 85         
        a = 5            
        delta_st = 0.2 * dev    
        delta_en = dev - delta_st 
        w = 2 * np.pi / dev     
        radius = 20      
        alpha = 24      
        
        # 회전교차로 중심 좌표 
        center = [merge_end_0[0] + access, merge_end_0[1]]
        radii = [radius, radius + 4]  
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [[c, s], [n, c]]  
        
        self.roundabout_center = center
        self.roundabout_radius = radius
        
        # hw_c 끝점에서 회전교차로 서쪽 입구까지의 직선 연결 차선
        roundabout_west_entry_0 = [center[0] - access, merge_end_0[1]]  # 상단 차선 진입점
        roundabout_west_entry_1 = [center[0] - access, merge_end_1[1]]  # 하단 차선 진입점

        # 직선 연결 차선 생성 (hw_c -> 회전교차로 진입 준비)
        if np.allclose(merge_end_0, roundabout_west_entry_0):
            roundabout_west_entry_0[0] += 0.1
        if np.allclose(merge_end_1, roundabout_west_entry_1):
            roundabout_west_entry_1[0] += 0.1

        net.add_lane("hw_c", "wer0", StraightLane(
            merge_end_0, roundabout_west_entry_0, line_types=line[0],
        ))
        net.add_lane("hw_c", "wer1", StraightLane(
            merge_end_1, roundabout_west_entry_1, line_types=line[1],
        ))

        # 8개 노드로 구성된 원형: se(남동) -> ex(동출구) -> ee(동입구) -> nx(북출구) 
        #                      -> ne(북입구) -> wx(서출구) -> we(서입구) -> sx(남출구) -> se
        for lane in [0, 1]:  
            # se -> ex: 남동에서 동쪽 출구로
            net.add_lane("se", "ex", CircularLane(
                center, radii[lane], np.deg2rad(90 - alpha), np.deg2rad(alpha),
                clockwise=False, line_types=line[lane],
            ))
            # ex -> ee: 동쪽 출구에서 동쪽 입구로  
            net.add_lane("ex", "ee", CircularLane(
                center, radii[lane], np.deg2rad(alpha), np.deg2rad(-alpha),
                clockwise=False, line_types=line[lane],
            ))
            # ee -> nx: 동쪽 입구에서 북쪽 출구로
            net.add_lane("ee", "nx", CircularLane(
                center, radii[lane], np.deg2rad(-alpha), np.deg2rad(-90 + alpha),
                clockwise=False, line_types=line[lane],
            ))
            # nx -> ne: 북쪽 출구에서 북쪽 입구로
            net.add_lane("nx", "ne", CircularLane(
                center, radii[lane], np.deg2rad(-90 + alpha), np.deg2rad(-90 - alpha),
                clockwise=False, line_types=line[lane],
            ))
            # ne -> wx: 북쪽 입구에서 서쪽 출구로
            net.add_lane("ne", "wx", CircularLane(
                center, radii[lane], np.deg2rad(-90 - alpha), np.deg2rad(-180 + alpha),
                clockwise=False, line_types=line[lane],
            ))
            # wx -> we: 서쪽 출구에서 서쪽 입구로
            net.add_lane("wx", "we", CircularLane(
                center, radii[lane], np.deg2rad(-180 + alpha), np.deg2rad(-180 - alpha),
                clockwise=False, line_types=line[lane],
            ))
            # we -> sx: 서쪽 입구에서 남쪽 출구로
            net.add_lane("we", "sx", CircularLane(
                center, radii[lane], np.deg2rad(-180 - alpha), np.deg2rad(-270 + alpha),
                clockwise=False, line_types=line[lane],
            ))
            # sx -> se: 남쪽 출구에서 남동으로 (순환 완성)
            net.add_lane("sx", "se", CircularLane(
                center, radii[lane], np.deg2rad(-270 + alpha), np.deg2rad(-270 - alpha),
                clockwise=False, line_types=line[lane],
            ))
        # 각 방향별로 직선 접근 + 사인파 곡선 + 원형 연결 구조
        
        # 남쪽 방향 (South) 진입/진출
        net.add_lane("ser", "ses", StraightLane([center[0] + 2, center[1] + access], [center[0] + 2, center[1] + dev / 2], line_types=(s, c)))
        net.add_lane("ses", "se", SineLane([center[0] + 2 + a, center[1] + dev / 2], [center[0] + 2 + a, center[1] + dev / 2 - delta_st], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("sx", "sxs", SineLane([center[0] - 2 - a, center[1] - dev / 2 + delta_en], [center[0] - 2 - a, center[1] + dev / 2], a, w, -np.pi / 2 + w * delta_en, line_types=(c, c)))
        net.add_lane("sxs", "sxr", StraightLane([center[0] - 2, center[1] + dev / 2], [center[0] - 2, center[1] + access], line_types=(n, c)))
        
        # 동쪽 방향 (East) 진입/진출
        net.add_lane("eer", "ees", StraightLane([center[0] + access, center[1] - 2], [center[0] + dev / 2, center[1] - 2], line_types=(s, c)))
        net.add_lane("ees", "ee", SineLane([center[0] + dev / 2, center[1] - 2 - a], [center[0] + dev / 2 - delta_st, center[1] - 2 - a], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("ex", "exs", SineLane([center[0] - dev / 2 + delta_en, center[1] + 2 + a], [center[0] + dev / 2, center[1] + 2 + a], a, w, -np.pi / 2 + w * delta_en, line_types=(c, c)))
        net.add_lane("exs", "exr", StraightLane([center[0] + dev / 2, center[1] + 2], [center[0] + access, center[1] + 2], line_types=(n, c)))
        
        # 북쪽 방향 (North) 진입/진출
        net.add_lane("ner", "nes", StraightLane([center[0] - 2, center[1] - access], [center[0] - 2, center[1] - dev / 2], line_types=(s, c)))
        net.add_lane("nes", "ne", SineLane([center[0] - 2 - a, center[1] - dev / 2], [center[0] - 2 - a, center[1] - dev / 2 + delta_st], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("nx", "nxs", SineLane([center[0] + 2 + a, center[1] + dev / 2 - delta_en], [center[0] + 2 + a, center[1] - dev / 2], a, w, -np.pi / 2 + w * delta_en, line_types=(c, c)))
        net.add_lane("nxs", "nxr", StraightLane([center[0] + 2, center[1] - dev / 2], [center[0] + 2, center[1] - access], line_types=(n, c)))
        
        # 서쪽 방향 (West) 진입/진출 - 고속도로에서 연결되는 주요 진입점
        net.add_lane("wer0", "wes", StraightLane(
            [center[0] - access, merge_end_0[1]],      
            [center[0] - dev / 2, merge_end_0[1]],
            line_types=(s, c)
        ))
        net.add_lane("wes", "we", SineLane(
            [center[0] - dev / 2, merge_end_0[1] + a],
            [center[0] - dev / 2 + delta_st, merge_end_0[1] + a],
            a, w, -np.pi / 2, line_types=(c, c)
        ))
        net.add_lane("wer1", "wxs0", StraightLane(
            [center[0] - access, merge_end_1[1]],      
            [center[0] - dev / 2, merge_end_1[1]],
            line_types=(n, c)
        ))
        net.add_lane("wxs0", "we", SineLane(
            [center[0] - dev / 2, merge_end_1[1] + a],
            [center[0] - dev / 2 + delta_st, merge_end_1[1] + a],
            a, w, -np.pi / 2, line_types=(c, c)
        ))

        # ================================================================
        # 4. 출구 구간 (Exit Section) - 목표 달성을 위한 탈출 경로
        # ================================================================        
        completion_distance = self.config.get("completion_distance", 50)
    
        net.add_lane("nxr", "north_exit", StraightLane(
            [center[0] + 2, center[1] - access], 
            [center[0] + 2, center[1] - access - completion_distance], 
            line_types=(n, c)
        ))
        
        net.add_lane("sxr", "south_exit", StraightLane(
            [center[0] - 2, center[1] + access], 
            [center[0] - 2, center[1] + access + completion_distance], 
            line_types=(n, c)
        ))

        net.add_lane("exr", "east_exit", StraightLane(
            [center[0] + access, center[1] + 2], 
            [center[0] + access + completion_distance, center[1] + 2], 
            line_types=(n, c)
        ))

        # ================================================================
        # 5. 구간별 라벨링 
        # ================================================================
       
        self.segment_labels = {
            # 고속도로 구간
            ("hw_a", "hw_b"): "highway_1",          
            ("hw_b", "hw_c"): "highway_2",          
            ("hw_c", "wer0"): "highway_to_roundabout", 
            ("hw_c", "wer1"): "highway_to_roundabout",
            
            # 합류 구간  
            ("mg_j", "mg_k"): "merge_straight",     
            ("mg_k", "hw_b"): "merge_entry",        

            # 회전교차로 진입 구간
            ("wer0", "wes"): "roundabout_entry",     
            ("wer1", "wxs0"): "roundabout_entry",
            ("wes", "we"): "roundabout_entry",      
            ("wxs0", "we"): "roundabout_entry",

            # 회전교차로 내부 순환 구간
            ("se", "ex"): "roundabout_internal",    
            ("ex", "ee"): "roundabout_internal",
            ("ee", "nx"): "roundabout_internal", 
            ("nx", "ne"): "roundabout_internal",
            ("ne", "wx"): "roundabout_internal",
            ("wx", "we"): "roundabout_internal",
            ("we", "sx"): "roundabout_internal",
            ("sx", "se"): "roundabout_internal",
            
            # 회전교차로 출구 구간
            ("sx", "sxs"): "roundabout_exit",        
            ("sxs", "sxr"): "roundabout_exit",
            ("ex", "exs"): "roundabout_exit",        
            ("exs", "exr"): "roundabout_exit",
            ("nx", "nxs"): "roundabout_exit",        
            ("nxs", "nxr"): "roundabout_exit",
            
            # 최종 탈출 구간
            ("nxr", "north_exit"): "final_exit",    
            ("sxr", "south_exit"): "final_exit",    
            ("exr", "east_exit"): "final_exit",      
        }   
        
        # 목표 출구 정보 저장 (종료 조건에서 사용)
        target_exit = self.config.get("roundabout_exit_target", "north")
        self.target_exit_lanes = {
            "north": [("nx", "nxs"), ("nxs", "nxr"), ("nxr", "north_exit")],
            "south": [("sx", "sxs"), ("sxs", "sxr"), ("sxr", "south_exit")],
            "east": [("ex", "exs"), ("exs", "exr"), ("exr", "east_exit")],
        }
        self.current_target_lanes = self.target_exit_lanes.get(target_exit, self.target_exit_lanes["north"])

        # 도로 네트워크 생성 및 등록
        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _get_segment_config(self, segment_type: str) -> dict:
        if segment_type in self.config["segment_configs"]:
            return self.config["segment_configs"][segment_type]
        
        for key in self.config["segment_configs"]:
            if segment_type.startswith(key):
                return self.config["segment_configs"][key]

        return self.config["segment_configs"].get("default", {})

    def _reset(self):
        self._make_road()
        self._make_vehicles()
    
    def _make_vehicles(self) -> None:
        road = self.road
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        # ================================================================
        # 1. 고속도로+합류 구간 차량 배치 (기존 로직)
        # ================================================================
        ego_vehicle = self.action_type.vehicle_class(
            road, road.network.get_lane(("hw_a", "hw_b", 1)).position(30, 0), speed=30
        )
        road.vehicles.append(ego_vehicle)
        merging_vehicle = other_vehicles_type(
            road, road.network.get_lane(("mg_j", "mg_k", 0)).position(110, 0), speed=20
        )
        merging_vehicle.target_speed = 30
        merging_vehicle.randomize_behavior()
        road.vehicles.append(merging_vehicle)

        for position, speed in [(90, 29), (70, 31), (5, 31.5)]:
            lane = road.network.get_lane(("hw_a", "hw_b", self.np_random.integers(2)))
            pos = lane.position(position + self.np_random.uniform(-5, 5), 0)
            spd = speed + self.np_random.uniform(-1, 1)
            v = other_vehicles_type(road, pos, speed=spd)
            v.randomize_behavior()
            road.vehicles.append(v)

        self.vehicle = ego_vehicle
        self.controlled_vehicles = [ego_vehicle]

        # 고속도로 구간 무작위 차량 생성 (수량 줄임)
        reduced_vehicle_count = max(5, self.config["vehicles_count"] // 10)  
        for _ in range(reduced_vehicle_count):
            try:
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=2 / self.config["vehicles_density"]  
                )
                vehicle.randomize_behavior()
                road.vehicles.append(vehicle)
            except Exception as e:
                print(f"무작위 차량 생성 실패: {e}")
                break

        # ================================================================
        # 2. 회전교차로 구간 차량 배치 
        # ================================================================
        
        roundabout_vehicles_created = 0
        
        print("도로 네트워크 구조 확인:")

        print("네트워크 노드들:")
        for from_node in road.network.graph.keys():
            connections = road.network.graph[from_node]
            print(f"  {from_node} -> {list(connections.keys())}")
        
        # 실제 차선 접근 방식 - (from_node, to_node, lane_index) 형태
        print("\n실제 차선들:")
        actual_lanes = []
        for from_node in road.network.graph.keys():
            for to_node in road.network.graph[from_node].keys():
                # road.network.graph[from_node][to_node]가 리스트인 경우 처리
                lanes_data = road.network.graph[from_node][to_node]
                if isinstance(lanes_data, list):
                    for lane_index in range(len(lanes_data)):
                        lane_key = (from_node, to_node, lane_index)
                        actual_lanes.append(lane_key)
                        print(f"  {lane_key}")
                elif isinstance(lanes_data, dict):
                    for lane_index in lanes_data.keys():
                        lane_key = (from_node, to_node, lane_index)
                        actual_lanes.append(lane_key)
                        print(f"  {lane_key}")
                else:
                    # 단일 차선인 경우
                    lane_key = (from_node, to_node, 0)
                    actual_lanes.append(lane_key)
                    print(f"  {lane_key}")
        
        print(f"\n총 실제 차선 수: {len(actual_lanes)}")

        roundabout_lanes = []
        for lane_key in actual_lanes:
            from_node, to_node, lane_index = lane_key
            if any(node in ["se", "ex", "ee", "nx", "ne", "wx", "we", "sx"] for node in [from_node, to_node]):
                roundabout_lanes.append(lane_key)
        
        print(f"회전교차로 관련 차선: {len(roundabout_lanes)}개")
        for lane_key in roundabout_lanes[:5]:
            print(f"  {lane_key}")
        
        # 회전교차로 차량 배치
        position_deviation = 2
        speed_deviation = 2
        destinations = ["exr", "sxr", "nxr"] 
        
        # 1. 주요 순환 차량 (we -> sx, lane 1) - 원본과 동일
        try:
            if ("we", "sx", 1) in [(key[0], key[1], key[2]) for key in actual_lanes]:
                vehicle = other_vehicles_type.make_on_lane(
                    self.road,
                    ("we", "sx", 1),
                    longitudinal=5 + self.np_random.normal() * position_deviation,
                    speed=16 + self.np_random.normal() * speed_deviation,
                )
                # 목적지 설정
                if hasattr(vehicle, 'plan_route_to'):
                    destination = self.np_random.choice(destinations)
                    try:
                        vehicle.plan_route_to(destination)
                    except:
                        pass
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)
                roundabout_vehicles_created += 1
                print(f"주요 순환 차량 생성 성공: we->sx (lane 1)")
            else:
                print("주요 순환 차선 없음: ('we', 'sx', 1)")
        except Exception as e:
            print(f"주요 순환 차량 생성 실패: {e}")

        # 2. 추가 순환 차량들 (we -> sx, lane 0) - 원본과 동일
        for i in list(range(1, 2)) + list(range(-1, 0)):
            try:
                if ("we", "sx", 0) in [(key[0], key[1], key[2]) for key in actual_lanes]:
                    vehicle = other_vehicles_type.make_on_lane(
                        self.road,
                        ("we", "sx", 0),
                        longitudinal=20 * i + self.np_random.normal() * position_deviation,
                        speed=16 + self.np_random.normal() * speed_deviation,
                    )
                    # 목적지 설정
                    if hasattr(vehicle, 'plan_route_to'):
                        try:
                            vehicle.plan_route_to(self.np_random.choice(destinations))
                        except:
                            pass
                    vehicle.randomize_behavior()
                    self.road.vehicles.append(vehicle)
                    roundabout_vehicles_created += 1
                    print(f"추가 순환 차량 {i} 생성 성공: we->sx (lane 0)")
                else:
                    print("추가 순환 차선 없음: ('we', 'sx', 0)")
            except Exception as e:
                print(f"추가 순환 차량 {i} 생성 실패: {e}")

        # 3. 동쪽 진입 차량 (eer -> ees) - 원본과 동일
        try:
            if ("eer", "ees", 0) in [(key[0], key[1], key[2]) for key in actual_lanes]:
                vehicle = other_vehicles_type.make_on_lane(
                    self.road,
                    ("eer", "ees", 0),
                    longitudinal=50 + self.np_random.normal() * position_deviation,
                    speed=16 + self.np_random.normal() * speed_deviation,
                )
                # 목적지 설정
                if hasattr(vehicle, 'plan_route_to'):
                    try:
                        vehicle.plan_route_to(self.np_random.choice(destinations))
                    except:
                        pass
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)
                roundabout_vehicles_created += 1
                print(f"동쪽 진입 차량 생성 성공: eer->ees")
            else:
                print("동쪽 진입 차선 없음: ('eer', 'ees', 0)")
        except Exception as e:
            print(f"동쪽 진입 차량 생성 실패: {e}")

        # 디버깅: 원본에서 사용하는 차선들 확인
        roundabout_target_lanes = [("we", "sx", 1), ("we", "sx", 0), ("eer", "ees", 0)]
        print(f"\n원본 roundabout 타겟 차선 존재 여부:")
        for target_lane in roundabout_target_lanes:
            exists = target_lane in [(key[0], key[1], key[2]) for key in actual_lanes]
            print(f"  {target_lane}: {'존재' if exists else '없음'}")

        print(f"\n총 {len(self.road.vehicles)}대 차량 배치 완료")
        print(f"- Ego 차량: 1대 (고속도로 시작)")
        print(f"- 회전교차로 차량: {roundabout_vehicles_created}대")
        print(f"- 기타 차량: {len(self.road.vehicles) - 1 - roundabout_vehicles_created}대")

    def _reward(self, action) -> float:
        rewards = self._rewards(action)

        lane_index = self.vehicle.lane_index
        if not lane_index:
            return 0.0

        _from, _to, _ = lane_index
        segment_key = (_from, _to)
        segment_type = self.segment_labels.get(segment_key, "default")
        segment_config = self._get_segment_config(segment_type)

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
            if abs(max_reward - min_reward) > 1e-8:
                reward = utils.lmap(reward, [min_reward, max_reward], [0, 1])

        # highway에서만 on_road_reward 곱하기
        if segment_type.startswith("highway") and "on_road_reward" in rewards:
            reward *= rewards["on_road_reward"]

        print(f"▶▶ Step {self.time} ◀◀")
        print(f"Segment: {segment_type}")
        print(f"Segment Config: {segment_config}")

        return reward
   
    
    def _rewards(self, action) -> dict[str, float]:
        """구간별 특성에 맞는 보상 계산"""
        lane_index = self.vehicle.lane_index
        if not lane_index:
            return {}

        # 차량 상태 유효성 검사
        if not hasattr(self.vehicle, 'position') or self.vehicle.position is None:
            return {}
        
        if not hasattr(self.vehicle, 'speed') or self.vehicle.speed is None:
            return {}

        _from, _to, _ = lane_index
        segment_key = (_from, _to)
        segment_type = self.segment_labels.get(segment_key, "default")
        
        # 라벨 기반 구간 분류
        if segment_type in ["highway_1", "highway_2", "highway_to_roundabout"]:
            main_segment = "highway"
        elif segment_type in ["merge_straight", "merge_entry"]:
            main_segment = "merge"
        elif segment_type in ["roundabout_entry", "roundabout_internal"]:
            main_segment = "roundabout"
        else:
            main_segment = "default"

        segment_config = self._get_segment_config(main_segment)
        rewards = {}

        # ================================================================
        # 공통 보상 요소
        # ================================================================
        # 충돌 보상 (모든 구간 공통)
        if "collision_reward" in segment_config:
            rewards["collision_reward"] = float(self.vehicle.crashed)

        # ================================================================
        # 고속도로 구간 보상: 빠르게 직진, 속도 중심, 차선 이탈 패널티
        # ================================================================
        if main_segment == "highway":
            # 고속 주행 보상 (주요 목표)
            if "high_speed_reward" in segment_config:
                try:
                    forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
                    speed_range = segment_config.get("reward_speed_range", [25, 35])
                    scaled_speed = utils.lmap(forward_speed, speed_range, [0, 1])
                    rewards["high_speed_reward"] = np.clip(scaled_speed, 0, 1)
                except (AttributeError, TypeError, ValueError):
                    rewards["high_speed_reward"] = 0.0

            # 우측 차선 유지 보상
            if "right_lane_reward" in segment_config:
                try:
                    neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
                    lane = (
                        self.vehicle.target_lane_index[2]
                        if isinstance(self.vehicle, ControlledVehicle)
                        else self.vehicle.lane_index[2]
                    )
                    rewards["right_lane_reward"] = lane / max(len(neighbours) - 1, 1)
                except (AttributeError, TypeError, ValueError, ZeroDivisionError):
                    rewards["right_lane_reward"] = 0.0

            # 도로 내 유지 보상
            if "on_road_reward" in segment_config:
                try:
                    rewards["on_road_reward"] = float(self.vehicle.on_road)
                except (AttributeError, TypeError):
                    rewards["on_road_reward"] = 0.0
            
            # 차선 이탈 패널티
            if "off_road_penalty" in segment_config:
                try:
                    rewards["off_road_penalty"] = float(not self.vehicle.on_road)
                except (AttributeError, TypeError):
                    rewards["off_road_penalty"] = 0.0
            
            # 차선 변경 보상 (상황에 맞는 적절한 차선 변경)
            if "lane_change_reward" in segment_config:
                lane_change_reward = 0.0
                if action in [0, 2]:  # LANE_LEFT, LANE_RIGHT
                    # 앞차와의 거리가 가까워서 추월이 필요한 상황
                    front_vehicle, _ = self.road.neighbour_vehicles(self.vehicle)
                    if front_vehicle and hasattr(front_vehicle, 'position') and front_vehicle.position is not None:
                        distance = front_vehicle.position[0] - self.vehicle.position[0]
                        if distance < 25:  # 25m 이내에 앞차가 있으면
                            lane_change_reward = 1.0  # 높은 보상
                        elif distance < 50:  # 50m 이내면 중간 보상
                            lane_change_reward = 0.5
                    else:
                        # 앞차가 없는 상황에서의 불필요한 차선 변경은 작은 패널티
                        lane_change_reward = -0.1
                
                rewards["lane_change_reward"] = lane_change_reward
                
            # 추월 보상 (새로 추가)
            if "overtaking_reward" in segment_config:
                try:
                    overtaking_reward = 0.0
                    front_vehicle, _ = self.road.neighbour_vehicles(self.vehicle)
                    if front_vehicle and hasattr(front_vehicle, 'position') and front_vehicle.position is not None:
                        distance = front_vehicle.position[0] - self.vehicle.position[0]
                        # 성공적인 추월 상황 (앞차를 추월한 경우)
                        if hasattr(self.vehicle, '_last_front_vehicle_distance'):
                            if (self.vehicle._last_front_vehicle_distance > 0 and 
                                distance < -5):  # 앞차를 추월했음
                                overtaking_reward = 1.0
                        self.vehicle._last_front_vehicle_distance = distance
                    
                    rewards["overtaking_reward"] = overtaking_reward
                except (AttributeError, TypeError, ValueError):
                    rewards["overtaking_reward"] = 0.0
                    
            # 안전한 거리 유지 보상 (새로 추가)  
            if "safe_following_reward" in segment_config:
                try:
                    front_vehicle, _ = self.road.neighbour_vehicles(self.vehicle)
                    if front_vehicle and hasattr(front_vehicle, 'position') and front_vehicle.position is not None:
                        distance = front_vehicle.position[0] - self.vehicle.position[0]
                        # 적절한 거리 유지 (15-30m)
                        if 15 <= distance <= 30:
                            rewards["safe_following_reward"] = 1.0
                        elif 10 <= distance < 15 or 30 < distance <= 40:
                            rewards["safe_following_reward"] = 0.5
                        else:
                            rewards["safe_following_reward"] = 0.0
                    else:
                        rewards["safe_following_reward"] = 0.0
                except (AttributeError, TypeError, ValueError):
                    rewards["safe_following_reward"] = 0.0

        # ================================================================
        # 합류 구간 보상: 우측 진입차량과 병합, 안전거리 유지, 차선 변경 보상
        # ================================================================
        elif main_segment == "merge":
            # 안전거리 유지 보상 (주요 목표)
            if "safe_distance_reward" in segment_config:
                try:
                    front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self.vehicle)
                    safe_distance = 0.0
                    
                    if front_vehicle and hasattr(front_vehicle, 'position') and front_vehicle.position is not None:
                        distance = front_vehicle.position[0] - self.vehicle.position[0]
                        safe_distance += np.clip(distance / 20.0, 0, 1)  # 20m 기준
                    if rear_vehicle and hasattr(rear_vehicle, 'position') and rear_vehicle.position is not None:
                        distance = self.vehicle.position[0] - rear_vehicle.position[0]
                        safe_distance += np.clip(distance / 15.0, 0, 1)  # 15m 기준
                    
                    rewards["safe_distance_reward"] = safe_distance / (2 if front_vehicle and rear_vehicle else 1)
                except (AttributeError, TypeError, ValueError, ZeroDivisionError):
                    rewards["safe_distance_reward"] = 0.0

            # 합류 협력 보상 (다른 차량 고려)
            if "merge_cooperation_reward" in segment_config:
                try:
                    merge_vehicles = [
                        v for v in self.road.vehicles 
                        if (hasattr(v, 'lane_index') and v.lane_index and 
                            self.segment_labels.get(v.lane_index[:2], "").startswith("merge"))
                    ]
                    cooperation_score = 0.0
                    for vehicle in merge_vehicles:
                        if (vehicle != self.vehicle and hasattr(vehicle, 'speed') and 
                            vehicle.speed is not None):
                            relative_speed = abs(vehicle.speed - self.vehicle.speed)
                            cooperation_score += np.exp(-relative_speed / 10.0)  # 속도 차이가 적을수록 높은 점수
                    
                    rewards["merge_cooperation_reward"] = cooperation_score / max(len(merge_vehicles) - 1, 1)
                except (AttributeError, TypeError, ValueError, ZeroDivisionError):
                    rewards["merge_cooperation_reward"] = 0.0

            # 적절한 차선 변경 보상
            if "lane_change_reward" in segment_config:
                rewards["lane_change_reward"] = float(action in [0, 2])

            # 속도 조절 보상
            if "speed_adaptation_reward" in segment_config:
                try:
                    speed_range = segment_config.get("reward_speed_range", [20, 30])
                    scaled_speed = utils.lmap(self.vehicle.speed, speed_range, [0, 1])
                    rewards["speed_adaptation_reward"] = np.clip(scaled_speed, 0, 1)
                except (AttributeError, TypeError, ValueError):
                    rewards["speed_adaptation_reward"] = 0.0

            # 합류 방해 패널티
            if "blocking_penalty" in segment_config:
                try:
                    blocking_score = 0.0
                    for vehicle in self.road.vehicles:
                        if (vehicle != self.vehicle and hasattr(vehicle, 'lane_index') and 
                            vehicle.lane_index and hasattr(vehicle, 'position') and 
                            vehicle.position is not None and
                            self.segment_labels.get(vehicle.lane_index[:2], "").startswith("merge")):
                            # 다른 합류 차량을 방해하는지 확인
                            if (abs(vehicle.position[0] - self.vehicle.position[0]) < 10 and
                                abs(vehicle.position[1] - self.vehicle.position[1]) < 3):
                                blocking_score += 1.0
                    
                    rewards["blocking_penalty"] = blocking_score
                except (AttributeError, TypeError, ValueError):
                    rewards["blocking_penalty"] = 0.0

        # ================================================================
        # 회전교차로 구간 보상: 진입 후 목표 출구로 탈출, 차선 유지, 중앙 충돌 회피
        # ================================================================
        elif main_segment == "roundabout":
            # 차선 유지 보상 (주요 목표)
            if "lane_keeping_reward" in segment_config:
                try:
                    # 차선 중앙 유지 정도 계산
                    if hasattr(self.vehicle, 'lane') and self.vehicle.lane is not None:
                        lateral_position = self.vehicle.lane.local_coordinates(self.vehicle.position)[1]
                        lane_keeping_score = np.exp(-abs(lateral_position) / 2.0)  # 중앙에 가까울수록 높은 점수
                        rewards["lane_keeping_reward"] = lane_keeping_score
                    else:
                        rewards["lane_keeping_reward"] = 0.0
                except (AttributeError, TypeError, ValueError):
                    rewards["lane_keeping_reward"] = 0.0

            # 부드러운 회전 보상
            if "smooth_turning_reward" in segment_config:
                try:
                    # 각속도 기반 부드러운 회전 평가
                    angular_velocity = abs(self.vehicle.heading_rate) if hasattr(self.vehicle, 'heading_rate') else 0
                    smooth_score = np.exp(-angular_velocity / 0.5)  # 급격한 회전일수록 낮은 점수
                    rewards["smooth_turning_reward"] = smooth_score
                except (AttributeError, TypeError, ValueError):
                    rewards["smooth_turning_reward"] = 0.5  # 기본값

            # 진행 상황 보상 (회전교차로 내에서의 진전)
            if "progress_reward" in segment_config:
                try:
                    # 현재 위치 기반 진행률 계산 (간단한 각도 기반)
                    center_pos = [400, 0]  # 회전교차로 중심 (근사값)
                    vehicle_angle = np.arctan2(
                        self.vehicle.position[1] - center_pos[1],
                        self.vehicle.position[0] - center_pos[0]
                    )
                    # 각도 기반 진행률 (0~1)
                    progress = (vehicle_angle + np.pi) / (2 * np.pi)
                    rewards["progress_reward"] = progress
                except (AttributeError, TypeError, ValueError):
                    rewards["progress_reward"] = 0.0

            # 적정 속도 제어 보상
            if "speed_control_reward" in segment_config:
                try:
                    speed_range = segment_config.get("reward_speed_range", [15, 25])
                    scaled_speed = utils.lmap(self.vehicle.speed, speed_range, [0, 1])
                    rewards["speed_control_reward"] = np.clip(scaled_speed, 0, 1)
                except (AttributeError, TypeError, ValueError):
                    rewards["speed_control_reward"] = 0.0

            # 중앙 충돌 패널티 (회전교차로 중앙 영역 침입)
            if "center_collision_penalty" in segment_config:
                try:
                    center_pos = [400, 0]  # 회전교차로 중심 (근사값)
                    distance_to_center = np.linalg.norm(
                        np.array(self.vehicle.position) - np.array(center_pos)
                    )
                    # 중앙 반지름 15m 이내 침입시 패널티
                    if distance_to_center < 15:
                        rewards["center_collision_penalty"] = 1.0
                    else:
                        rewards["center_collision_penalty"] = 0.0
                except (AttributeError, TypeError, ValueError):
                    rewards["center_collision_penalty"] = 0.0

            # 성공적 진입 보상 (진입 구간에서만)
            if "entry_success_reward" in segment_config and segment_type == "roundabout_entry":
                rewards["entry_success_reward"] = 1.0  # 진입 성공시 보상
            else:
                rewards["entry_success_reward"] = 0.0

        return rewards
    
    # 레이블을 반영하여 환경마다 종료 조건 반영
    def _is_terminated(self) -> bool:
        """구간별 종료 조건 - 회전교차로 목표 달성 중심"""
        # 기본 안전 검사
        if not hasattr(self.vehicle, 'lane_index') or not self.vehicle.lane_index:
            return False
            
        if not hasattr(self.vehicle, 'position') or self.vehicle.position is None:
            return False
            
        lane_index = self.vehicle.lane_index
        _from, _to, _ = lane_index
        segment_key = (_from, _to)
        segment_type = self.segment_labels.get(segment_key, "default")

        # 라벨 기반 구간 분류
        if segment_type in ["highway_1", "highway_2", "highway_to_roundabout"]:
            main_segment = "highway"
        elif segment_type in ["merge_straight", "merge_entry"]:
            main_segment = "merge"
        elif segment_type in ["roundabout_entry", "roundabout_internal"]:
            main_segment = "roundabout"
        elif segment_type == "final_exit":
            main_segment = "final_exit"
        else:
            main_segment = "default"

        # ================================================================
        # 구간별 종료 조건
        # ================================================================

        if main_segment == "final_exit":
            try:
                # 목표 출구에서 충분히 멀리 나갔는지 확인
                completion_distance = self.config.get("completion_distance", 50)
                
       
                target_exit = self.config.get("roundabout_exit_target", "north")
                center = self.roundabout_center
                
                if target_exit == "north" and segment_key == ("nxr", "north_exit"):
                  
                    if self.vehicle.position[1] <= center[1] - 170 - completion_distance * 0.8:
                        print(f" 목표 달성! 북쪽 출구로 성공적 탈출 완료!")
                        return True
                elif target_exit == "south" and segment_key == ("sxr", "south_exit"):
             
                    if self.vehicle.position[1] >= center[1] + 170 + completion_distance * 0.8:
                        print(f" 목표 달성! 남쪽 출구로 성공적 탈출 완료!")
                        return True
                elif target_exit == "east" and segment_key == ("exr", "east_exit"):
           
                    if self.vehicle.position[0] >= center[0] + 170 + completion_distance * 0.8:
                        print(f" 목표 달성! 동쪽 출구로 성공적 탈출 완료!")
                        return True
                
                if (segment_key in [("nxr", "north_exit"), ("sxr", "south_exit"), ("exr", "east_exit")] and
                    segment_key != (f"{target_exit[0]}xr", f"{target_exit}_exit")):
                    print(f" 부분 성공! {segment_key[1].replace('_exit', '')} 출구로 탈출 완료 (목표는 {target_exit})")
                    return True
                    
            except (AttributeError, TypeError, ValueError, IndexError):
                pass
        
        # 고속도로 구간: 충돌시에만 종료 
        if main_segment == "highway":
            try:
                return (self.vehicle.crashed or 
                        (self.config["offroad_terminal"] and not self.vehicle.on_road))
            except (AttributeError, TypeError):
                return False
        
        # 합류 구간: 충돌 or 정체
        elif main_segment == "merge":
            try:
                if self.vehicle.crashed:
                    return True
                    
                # 정체 상황 감지 
                if not hasattr(self, '_low_speed_counter'):
                    self._low_speed_counter = 0
                
                if hasattr(self.vehicle, 'speed') and self.vehicle.speed is not None:
                    if self.vehicle.speed < 5/3.6:  # 5km/h를 m/s로 변환
                        self._low_speed_counter += 1
                    else:
                        self._low_speed_counter = 0
                    
                    # 10초 이상 정체시 종료 (더 관대하게 조정)
                    if self._low_speed_counter > 10 * 10:  # 10Hz * 10초
                        print(f" 합류 구간에서 장시간 정체로 인한 실패")
                        return True
                        
            except (AttributeError, TypeError, IndexError):
                return False
        
        # 회전교차로 구간: 충돌, 중앙 침입, 심각한 역방향 주행
        elif main_segment == "roundabout":
            try:
                # 충돌시 종료
                if self.vehicle.crashed:
                    print(f" 회전교차로에서 충돌 발생")
                    return True
                
                # 회전교차로 중앙 충돌 (반지름 15m 이내 침입)
                center = self.roundabout_center
                distance_to_center = np.linalg.norm(
                    np.array(self.vehicle.position) - np.array(center)
                )
                if distance_to_center < 15:
                    print(f" 회전교차로 중앙 영역 침입으로 인한 실패")
                    return True
                
                # 회전교차로 영역을 완전히 벗어났지만 출구가 아닌 곳으로 나간 경우
                if distance_to_center > 200:
                    # 올바른 출구 차선에 있는지 확인
                    if segment_key not in [("sx", "sxs"), ("sxs", "sxr"), ("ex", "exs"), 
                                         ("exs", "exr"), ("nx", "nxs"), ("nxs", "nxr")]:
                        print(f" 회전교차로에서 잘못된 경로로 이탈")
                        return True
                
                if hasattr(self.vehicle, 'heading') and self.vehicle.heading is not None:
                    # 회전교차로에서 예상되는 방향 (반시계방향)
                    expected_heading = np.arctan2(
                        center[1] - self.vehicle.position[1],
                        center[0] - self.vehicle.position[0]
                    ) + np.pi/2  # 반시계방향
                    
                    heading_diff = abs(self.vehicle.heading - expected_heading)
                    if heading_diff > np.pi:
                        heading_diff = 2*np.pi - heading_diff
                    
                    # 135도 이상 방향이 틀렸을 때만 실패 
                    if heading_diff > 3*np.pi/4:
                        print(f" 회전교차로에서 심각한 역방향 주행")
                        return True
                        
            except (AttributeError, TypeError, ValueError, IndexError):
                return False
        
        return False
    
    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]
    
