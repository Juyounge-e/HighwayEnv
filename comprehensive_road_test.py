"""
도로 주행 가능성 종합 테스트
=========================

이 스크립트는 custom-mixed-road 환경에서 단독 에이전트가 
전체 도로 구간을 성공적으로 주행할 수 있는지 테스트합니다.

테스트 항목:
1. 도로 연결성 검증
2. 단독 주행 테스트 (다양한 액션 전략)
3. 구간별 주행 분석
4. 시각화 및 보고서 생성
5. 실시간 시각적 확인 및 비디오 저장
"""

import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter
import json
import time
import os
from typing import Dict, List, Tuple, Optional
import cv2

def setup_korean_font():
    """macOS에서 한글 폰트 설정"""
    try:
        # macOS에서 사용 가능한 한글 폰트 목록
        korean_fonts = [
            'Apple SD Gothic Neo',
            'Noto Sans CJK KR',
            'Malgun Gothic',
            'NanumGothic',
            'AppleGothic'
        ]
        
        # 시스템에 설치된 폰트 목록 가져오기
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        # 사용 가능한 한글 폰트 찾기
        for font in korean_fonts:
            if font in available_fonts:
                plt.rcParams['font.family'] = font
                plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
                print(f" 한글 폰트 설정 완료: {font}")
                return True
        
        # 한글 폰트를 찾지 못한 경우 기본 설정
        print(" 한글 폰트를 찾을 수 없어 기본 설정을 사용합니다.")
        plt.rcParams['axes.unicode_minus'] = False
        return False
        
    except Exception as e:
        print(f" 폰트 설정 중 오류 발생: {e}")
        plt.rcParams['axes.unicode_minus'] = False
        return False

# 한글 폰트 설정 실행
setup_korean_font()

class RoadTester:
    def __init__(self):
        self.env = None
        self.test_results = {}
        self.trajectory_data = []
        self.episode_frames = []  # 에피소드 프레임 저장
        
    def setup_environment(self, vehicles_count: int = 0, duration: int = 200):
        """환경 설정"""
        print(" 환경 설정 중...")
        
        self.env = gym.make("custom-mixed-road-v0", render_mode="rgb_array")
        
        # 단독 주행을 위한 설정
        self.env.unwrapped.configure({
            "vehicles_count": vehicles_count,           # 다른 차량 수
            "controlled_vehicles": 1,                   # 제어 차량 1대
            "duration": duration,                       # 시뮬레이션 시간
            "simulation_frequency": 15,                 # 시뮬레이션 주파수
            "policy_frequency": 5,                      # 정책 주파수
            "collision_reward": -10,                    # 충돌 패널티 강화
            "offroad_terminal": True,                   # 도로 이탈시 종료
            "normalize_reward": False,                  # 보상 정규화 비활성화
            "roundabout_exit_target": "north",          # 목표 출구
            "success_reward": 100.0,                    # 성공 보상
            "completion_distance": 50,                  # 완료 거리
        })
        
        print(f"환경 설정 완료 (차량 수: {vehicles_count}, 지속 시간: {duration})")
        
    def analyze_road_network(self) -> Dict:
        print("\n  도로 네트워크 분석 중...")
        
        obs, info = self.env.reset()
        road = self.env.unwrapped.road
        network = road.network
        
        # 네트워크 구조 분석
        nodes = list(network.graph.keys())
        connections = {}
        total_lanes = 0
        
        for from_node in network.graph.keys():
            connections[from_node] = list(network.graph[from_node].keys())
            for to_node in network.graph[from_node].keys():
                lanes_data = network.graph[from_node][to_node]
                if isinstance(lanes_data, list):
                    total_lanes += len(lanes_data)
                elif isinstance(lanes_data, dict):
                    total_lanes += len(lanes_data)
                else:
                    total_lanes += 1
        
        # 구간별 분류
        highway_segments = [key for key in connections.keys() if key.startswith('hw')]
        merge_segments = [key for key in connections.keys() if key.startswith('mg')]
        roundabout_segments = [key for key in connections.keys() if key in 
                             ['se', 'ex', 'ee', 'nx', 'ne', 'wx', 'we', 'sx']]
        exit_segments = [key for key in connections.keys() if 'exit' in key]
        
        analysis = {
            "total_nodes": len(nodes),
            "total_lanes": total_lanes,
            "highway_segments": len(highway_segments),
            "merge_segments": len(merge_segments),
            "roundabout_segments": len(roundabout_segments),
            "exit_segments": len(exit_segments),
            "connections": connections,
            "nodes": nodes
        }
        
        print(f" 네트워크 분석 결과:")
        print(f"   - 총 노드 수: {analysis['total_nodes']}")
        print(f"   - 총 차선 수: {analysis['total_lanes']}")
        print(f"   - 고속도로 구간: {analysis['highway_segments']}")
        print(f"   - 합류 구간: {analysis['merge_segments']}")
        print(f"   - 회전교차로 구간: {analysis['roundabout_segments']}")
        print(f"   - 출구 구간: {analysis['exit_segments']}")
        
        return analysis
    
    def test_driving_strategy(self, strategy_name: str, action_sequence: List[int], 
                            max_steps: int = 1000, enable_visual: bool = False, 
                            save_video: bool = False) -> Dict:
        """특정 주행 전략 테스트"""
        print(f"\n 주행 전략 테스트: {strategy_name}")
        if enable_visual:
            print("     시각적 모니터링 활성화")
        if save_video:
            print("    비디오 저장 활성화")
        
        obs, info = self.env.reset()
        
        trajectory = []
        segment_history = []
        reward_history = []
        frames = []  # 프레임 저장용
        
        done = False
        truncated = False
        step = 0
        action_idx = 0
        
        start_time = time.time()
        
        # 시각화 설정
        if enable_visual:
            plt.ion()  # 인터랙티브 모드 활성화
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.set_title(f'실시간 주행 모니터링: {strategy_name}')
        
        while not (done or truncated) and step < max_steps:
            # 액션 선택 (순환 또는 고정)
            if action_sequence:
                action = action_sequence[action_idx % len(action_sequence)]
            else:
                action = 1  # IDLE
            
            # 스텝 실행
            obs, reward, done, truncated, info = self.env.step(action)
            
            # 프레임 캡처 (비디오 저장용)
            if save_video or enable_visual:
                frame = self.env.render()
                if frame is not None:
                    frames.append(frame)
            
            # 차량 상태 기록
            ego = self.env.unwrapped.vehicle
            if ego and hasattr(ego, 'position') and ego.position is not None:
                position = ego.position.copy()
                speed = ego.speed if hasattr(ego, 'speed') else 0
                crashed = ego.crashed if hasattr(ego, 'crashed') else False
                on_road = ego.on_road if hasattr(ego, 'on_road') else True
                
                # 현재 구간 정보
                current_segment = "unknown"
                if hasattr(ego, 'lane_index') and ego.lane_index:
                    lane_key = ego.lane_index[:2]
                    segment_labels = getattr(self.env.unwrapped, 'segment_labels', {})
                    current_segment = segment_labels.get(lane_key, f"{lane_key[0]}->{lane_key[1]}")
                
                trajectory.append({
                    'step': step,
                    'position': position,
                    'speed': speed,
                    'crashed': crashed,
                    'on_road': on_road,
                    'reward': reward,
                    'action': action,
                    'segment': current_segment
                })
                
                segment_history.append(current_segment)
                reward_history.append(reward)
                
                # 실시간 시각화 업데이트
                if enable_visual and step % 5 == 0:  # 5스텝마다 업데이트
                    ax.clear()
                    if len(frames) > 0:
                        ax.imshow(frames[-1])
                        ax.set_title(f'Step {step}: {current_segment} | Speed: {speed:.1f} | Reward: {reward:.2f}')
                        ax.axis('off')
                        plt.pause(0.01)
                
                # 진행 상황 출력 (10스텝마다)
                if step % 10 == 0:
                    print(f"   Step {step:3d}: pos=({position[0]:6.1f}, {position[1]:6.1f}), "
                          f"speed={speed:5.1f}, segment={current_segment}, reward={reward:6.2f}")
            
            step += 1
            action_idx += 1

            if len(trajectory) > 20:
                recent_positions = [t['position'] for t in trajectory[-20:]]
                if all(np.linalg.norm(np.array(pos) - np.array(recent_positions[0])) < 2.0 
                       for pos in recent_positions):
                    print(f"     같은 위치에서 정체 감지, 테스트 중단")
                    break
        
        # 시각화 정리
        if enable_visual:
            plt.ioff()
            plt.close(fig)
        
        end_time = time.time()
        
        # 비디오 저장
        if save_video and frames:
            self._save_video(frames, f"{strategy_name}_driving.mp4")
            self.episode_frames = frames  # 마지막 에피소드 프레임 저장
        
        # 결과 분석
        final_position = trajectory[-1]['position'] if trajectory else [0, 0]
        total_distance = 0
        if len(trajectory) > 1:
            for i in range(1, len(trajectory)):
                dist = np.linalg.norm(
                    np.array(trajectory[i]['position']) - np.array(trajectory[i-1]['position'])
                )
                total_distance += dist
        
        unique_segments = list(set(segment_history))
        segment_counts = {seg: segment_history.count(seg) for seg in unique_segments}
        
        result = {
            'strategy_name': strategy_name,
            'success': done and not truncated and not trajectory[-1]['crashed'] if trajectory else False,
            'steps_taken': step,
            'final_position': final_position,
            'total_distance': total_distance,
            'average_speed': np.mean([t['speed'] for t in trajectory]) if trajectory else 0,
            'total_reward': sum(reward_history),
            'crashed': trajectory[-1]['crashed'] if trajectory else True,
            'segments_visited': unique_segments,
            'segment_counts': segment_counts,
            'trajectory': trajectory,
            'duration_seconds': end_time - start_time,
            'frames_captured': len(frames) if save_video else 0
        }
        
        # 결과 출력
        status = " 성공" if result['success'] else " 실패"
        crash_status = "충돌" if result['crashed'] else "안전"
        
        print(f"   {status} - {step}스텝, 거리={total_distance:.1f}m, {crash_status}")
        print(f"   방문 구간: {len(unique_segments)}개 - {unique_segments}")
        if save_video:
            print(f"    {len(frames)}개 프레임 캡처됨")
        
        return result
    
    def _save_video(self, frames: List[np.ndarray], filename: str):
        """프레임들을 비디오 파일로 저장"""
        if not frames:
            return
            
        print(f"    비디오 저장 중: {filename}")
        
        try:
            # OpenCV를 사용한 비디오 저장
            height, width, layers = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(filename, fourcc, 10.0, (width, height))
            
            for frame in frames:
                # RGB to BGR 변환 (OpenCV는 BGR 사용)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video.write(frame_bgr)
            
            video.release()
            print(f"    비디오 저장 완료: {filename}")
            
        except Exception as e:
            print(f"    비디오 저장 실패: {e}")
            # 대안: matplotlib을 사용한 GIF 저장
            try:
                self._save_gif(frames, filename.replace('.mp4', '.gif'))
            except Exception as e2:
                print(f"   GIF 저장도 실패: {e2}")
    
    def _save_gif(self, frames: List[np.ndarray], filename: str):
        """프레임들을 GIF 파일로 저장"""
        print(f"    GIF 저장 중: {filename}")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        
        def animate(frame_idx):
            ax.clear()
            ax.imshow(frames[frame_idx])
            ax.axis('off')
            ax.set_title(f'Frame {frame_idx}/{len(frames)}')
        
        anim = FuncAnimation(fig, animate, frames=len(frames), interval=100, repeat=True)
        anim.save(filename, writer=PillowWriter(fps=10))
        plt.close(fig)
        print(f"    GIF 저장 완료: {filename}")
    
    def run_comprehensive_test(self, enable_visual: bool = False, save_videos: bool = False) -> Dict:
        """종합 테스트 실행"""
        print("\n 종합 도로 주행 테스트 시작")
        if enable_visual:
            print(" 시각적 모니터링 모드 활성화")
        if save_videos:
            print(" 비디오 저장 모드 활성화")
        print("=" * 50)
        
        # 1. 도로 네트워크 분석
        network_analysis = self.analyze_road_network()
        
        # 2. 다양한 주행 전략 테스트
        strategies = [
            ("직진만", [1]),                                    # IDLE만
            ("좌우_교대", [0, 1, 2, 1]),                        # 좌우 차선변경
            ("가속_직진", [3, 1]),                              # 가속 후 직진
            ("감속_직진", [4, 1]),                              # 감속 후 직진
            ("적응형", [1, 1, 1, 3, 1, 1, 2, 1, 1, 0]),        # 복합 전략
            ("안전_주행", [1, 1, 1, 1, 4, 1, 1, 1]),           # 안전 우선
        ]
        
        test_results = {}
        successful_strategy_found = False
        
        for strategy_name, action_sequence in strategies:
            try:
                # 성공한 전략이 발견되면 시각적 확인 활성화
                visual_mode = enable_visual or successful_strategy_found
                video_mode = save_videos and (successful_strategy_found or strategy_name == strategies[-1][0])
                
                result = self.test_driving_strategy(
                    strategy_name, action_sequence, max_steps=800,
                    enable_visual=visual_mode, save_video=video_mode
                )
                test_results[strategy_name] = result
                
                # 성공한 전략이 있으면 상세 분석
                if result['success']:
                    print(f" {strategy_name} 전략으로 완주 성공!")
                    self.trajectory_data = result['trajectory']
                    successful_strategy_found = True
                    
                    # 성공한 전략은 반드시 비디오 저장
                    if not video_mode and save_videos:
                        print("    성공한 전략 재실행하여 비디오 저장...")
                        success_result = self.test_driving_strategy(
                            f"{strategy_name}_SUCCESS", action_sequence, max_steps=800,
                            enable_visual=False, save_video=True
                        )
                    break
                    
            except Exception as e:
                print(f"    {strategy_name} 전략 테스트 중 오류: {e}")
                test_results[strategy_name] = {'error': str(e), 'success': False}
        
        # 3. 결과 종합
        successful_strategies = [name for name, result in test_results.items() 
                               if result.get('success', False)]
        
        comprehensive_result = {
            'network_analysis': network_analysis,
            'strategy_results': test_results,
            'successful_strategies': successful_strategies,
            'road_completable': len(successful_strategies) > 0,
            'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'visual_enabled': enable_visual,
            'videos_saved': save_videos
        }
        
        return comprehensive_result
    
    def create_episode_playback(self, save_path: str = "episode_playback.gif"):
        """마지막 에피소드의 재생 가능한 시각화 생성"""
        if not self.episode_frames:
            print(" 저장된 에피소드 프레임이 없습니다.")
            return
        
        print(f"\n 에피소드 재생 시각화 생성 중... (저장 경로: {save_path})")
        
        # 인터랙티브 재생기 생성
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 첫 번째 서브플롯: 에피소드 프레임
        ax1.set_title('주행 에피소드')
        ax1.axis('off')
        
        # 두 번째 서브플롯: 궤적과 통계
        ax2.set_title('주행 궤적 및 통계')
        
        def update_playback(frame_idx):
            # 프레임 표시
            ax1.clear()
            ax1.imshow(self.episode_frames[frame_idx])
            ax1.set_title(f'Frame {frame_idx+1}/{len(self.episode_frames)}')
            ax1.axis('off')
            
            # 궤적 업데이트
            if self.trajectory_data and frame_idx < len(self.trajectory_data):
                ax2.clear()
                
                # 현재까지의 궤적
                current_traj = self.trajectory_data[:frame_idx+1]
                positions = [t['position'] for t in current_traj]
                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]
                
                # 궤적 그리기
                ax2.plot(x_coords, y_coords, 'b-', alpha=0.6, linewidth=2, label='주행 궤적')
                
                # 현재 위치 강조
                if positions:
                    ax2.scatter(x_coords[-1], y_coords[-1], color='red', s=100, 
                              label='현재 위치', zorder=5)
                
                # 회전교차로 표시
                if hasattr(self.env.unwrapped, 'roundabout_center'):
                    center = self.env.unwrapped.roundabout_center
                    radius = self.env.unwrapped.roundabout_radius
                    circle = Circle(center, radius, fill=False, color='orange', linewidth=2)
                    ax2.add_patch(circle)
                
                # 현재 상태 정보
                current_data = current_traj[-1]
                info_text = f"""
Step: {current_data['step']}
Speed: {current_data['speed']:.1f} km/h
Segment: {current_data['segment']}
Reward: {current_data['reward']:.2f}
Status: {'Crashed' if current_data['crashed'] else 'OK'}
"""
                ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes, 
                        verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax2.set_xlabel('X 좌표 (m)')
                ax2.set_ylabel('Y 좌표 (m)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        # 애니메이션 생성
        anim = FuncAnimation(fig, update_playback, frames=len(self.episode_frames), 
                           interval=200, repeat=True)
        
        # 저장
        try:
            anim.save(save_path, writer=PillowWriter(fps=5))
            print(f" 에피소드 재생 시각화 저장 완료: {save_path}")
        except Exception as e:
            print(f" 에피소드 재생 시각화 저장 실패: {e}")
        
        plt.close(fig)
        return anim
    
    def visualize_results(self, results: Dict, save_path: str = "road_test_results.png"):
        print(f"\n 결과 시각화 중... (저장 경로: {save_path})")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('도로 주행 가능성 테스트 결과', fontsize=16, fontweight='bold')
        
        # 1. 네트워크 구조 (좌상)
        ax1 = axes[0, 0]
        network = results['network_analysis']
        categories = ['고속도로', '합류', '회전교차로', '출구']
        counts = [
            network['highway_segments'],
            network['merge_segments'], 
            network['roundabout_segments'],
            network['exit_segments']
        ]
        ax1.bar(categories, counts, color=['blue', 'green', 'orange', 'red'])
        ax1.set_title('도로 구간별 노드 수')
        ax1.set_ylabel('노드 수')
        
        # 2. 전략별 성공률 (중상)
        ax2 = axes[0, 1]
        strategy_names = list(results['strategy_results'].keys())
        success_rates = [1 if results['strategy_results'][name].get('success', False) else 0 
                        for name in strategy_names]
        colors = ['green' if success else 'red' for success in success_rates]
        ax2.bar(strategy_names, success_rates, color=colors)
        ax2.set_title('전략별 성공률')
        ax2.set_ylabel('성공 (1) / 실패 (0)')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. 전략별 주행 거리 (우상)
        ax3 = axes[0, 2]
        distances = [results['strategy_results'][name].get('total_distance', 0) 
                    for name in strategy_names]
        ax3.bar(strategy_names, distances, color=['lightblue'] * len(strategy_names))
        ax3.set_title('전략별 주행 거리')
        ax3.set_ylabel('거리 (m)')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. 주행 궤적 (좌하) - 성공한 경우만
        ax4 = axes[1, 0]
        if self.trajectory_data:
            positions = [t['position'] for t in self.trajectory_data]
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            # 궤적을 구간별로 색상 구분
            segments = [t['segment'] for t in self.trajectory_data]
            segment_colors = {
                'highway_1': 'blue', 'highway_2': 'lightblue', 'highway_to_roundabout': 'cyan',
                'merge_straight': 'green', 'merge_entry': 'lightgreen',
                'roundabout_entry': 'orange', 'roundabout_internal': 'red',
                'roundabout_exit': 'purple', 'final_exit': 'gold'
            }
            
            # 구간별 궤적 그리기
            current_segment = segments[0] if segments else 'unknown'
            segment_start = 0
            
            for i, segment in enumerate(segments + ['end']):  # 마지막 구간 처리를 위해 'end' 추가
                if segment != current_segment or i == len(segments):
                    # 현재 구간 그리기
                    if segment_start < len(x_coords):
                        end_idx = min(i, len(x_coords))
                        color = segment_colors.get(current_segment, 'gray')
                        ax4.plot(x_coords[segment_start:end_idx], y_coords[segment_start:end_idx], 
                                color=color, linewidth=2, alpha=0.7, label=current_segment)
                    
                    current_segment = segment
                    segment_start = i
            
            # 시작점과 종료점 표시
            ax4.scatter(x_coords[0], y_coords[0], color='green', s=100, label='시작점', zorder=5)
            ax4.scatter(x_coords[-1], y_coords[-1], color='red', s=100, label='종료점', zorder=5)
            
            # 회전교차로 표시
            if hasattr(self.env.unwrapped, 'roundabout_center'):
                center = self.env.unwrapped.roundabout_center
                radius = self.env.unwrapped.roundabout_radius
                circle = Circle(center, radius, fill=False, color='orange', linewidth=2)
                ax4.add_patch(circle)
            
            ax4.set_title('주행 궤적 (구간별 색상)')
            ax4.set_xlabel('X 좌표 (m)')
            ax4.set_ylabel('Y 좌표 (m)')
            ax4.grid(True, alpha=0.3)
            # 범례가 너무 많으면 생략
            handles, labels = ax4.get_legend_handles_labels()
            if len(handles) <= 8:
                ax4.legend(fontsize='small')
        else:
            ax4.text(0.5, 0.5, '성공한 주행 없음', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('주행 궤적 (없음)')
        
        # 5. 시간별 속도 변화 (중하)
        ax5 = axes[1, 1]
        if self.trajectory_data:
            steps = [t['step'] for t in self.trajectory_data]
            speeds = [t['speed'] for t in self.trajectory_data]
            ax5.plot(steps, speeds, 'b-', linewidth=2)
            ax5.set_title('시간별 속도 변화')
            ax5.set_xlabel('Step')
            ax5.set_ylabel('속도 (m/s)')
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, '데이터 없음', ha='center', va='center', 
                    transform=ax5.transAxes, fontsize=12)
            ax5.set_title('속도 변화 (없음)')
        
        # 6. 종합 결과 (우하)
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # 결과 텍스트
        result_text = f"""
종합 테스트 결과

 도로 완주 가능: {'예' if results['road_completable'] else '아니오'}

 네트워크 정보:
   • 총 노드: {network['total_nodes']}개
   • 총 차선: {network['total_lanes']}개

 테스트된 전략: {len(strategy_names)}개
   • 성공한 전략: {len(results['successful_strategies'])}개
   • 성공률: {len(results['successful_strategies'])/len(strategy_names)*100:.1f}%

 성공 전략:
"""
        
        for strategy in results['successful_strategies']:
            result_text += f"   • {strategy}\n"
        
        if not results['successful_strategies']:
            result_text += "   • 없음 (도로 연결 문제 가능성)\n"
        
        # 시각적 기능 사용 여부 표시
        if results.get('visual_enabled'):
            result_text += "\n  시각적 모니터링 사용됨"
        if results.get('videos_saved'):
            result_text += "\n 비디오 저장됨"
        
        result_text += f"\n 테스트 시간: {results['test_timestamp']}"
        
        ax6.text(0.05, 0.95, result_text, transform=ax6.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" 결과 시각화 완료: {save_path}")
        
        return fig

    def save_detailed_report(self, results: Dict, save_path: str = "road_test_report.json"):
        """상세 보고서 저장"""
        print(f"\n 상세 보고서 저장 중... (저장 경로: {save_path})")
        
        # JSON 직렬화를 위해 numpy 배열 변환
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        clean_results = convert_numpy(results)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2, ensure_ascii=False)
        
        print(f" 상세 보고서 저장 완료: {save_path}")
    
    def print_summary(self, results: Dict):
        """결과 요약 출력"""
        print("\n" + "="*60)
        print(" 도로 주행 가능성 테스트 최종 결과")
        print("="*60)
        
        if results['road_completable']:
            print(" 결론: 이 도로는 단독 에이전트로 완주 가능합니다!")
            print(f"   성공한 전략: {', '.join(results['successful_strategies'])}")
        else:
            print(" 결론: 현재 도로 구조에서는 완주가 어렵습니다.")
            print("   가능한 문제:")
            print("   • 도로 연결 불완전")
            print("   • 차선 변경 불가능 구간")
            print("   • 회전교차로 진입/진출 문제")
        
        print(f"\n 테스트 통계:")
        print(f"   • 테스트된 전략 수: {len(results['strategy_results'])}")
        print(f"   • 성공률: {len(results['successful_strategies'])}/{len(results['strategy_results'])}")
        print(f"   • 도로 구간 수: {results['network_analysis']['total_nodes']}")
        
        # 시각적 기능 사용 여부
        if results.get('visual_enabled'):
            print(f"   •   시각적 모니터링 사용됨")
        if results.get('videos_saved'):
            print(f"   •  비디오 파일 저장됨")
        
        # 실패한 전략들의 공통 실패 지점 분석
        failed_strategies = {name: result for name, result in results['strategy_results'].items() 
                           if not result.get('success', False)}
        
        if failed_strategies:
            print(f"\n 실패 분석:")
            for name, result in failed_strategies.items():
                if 'error' in result:
                    print(f"   • {name}: 오류 - {result['error']}")
                else:
                    crashed = result.get('crashed', True)
                    final_pos = result.get('final_position', [0, 0])
                    segments = result.get('segments_visited', [])
                    print(f"   • {name}: {'충돌' if crashed else '미완주'} "
                          f"at ({final_pos[0]:.1f}, {final_pos[1]:.1f}), "
                          f"구간 {len(segments)}개 방문")
        
        print("="*60)
    
    def cleanup(self):
        """리소스 정리"""
        if self.env:
            self.env.close()

def main():
    """메인 테스트 실행"""
    print(" 도로 주행 가능성 종합 테스트 시작")
    print("이 테스트는 custom-mixed-road 환경에서 단독 에이전트의 완주 가능성을 검증합니다.\n")
    
    # 사용자 옵션 (필요에 따라 수정)
    ENABLE_VISUAL = True    # 실시간 시각적 모니터링
    SAVE_VIDEOS = True      # 비디오 저장
    CREATE_PLAYBACK = True  # 에피소드 재생 시각화
    
    tester = RoadTester()
    
    try:
        # 환경 설정
        tester.setup_environment(vehicles_count=0, duration=300)
        
        # 종합 테스트 실행
        results = tester.run_comprehensive_test(
            enable_visual=ENABLE_VISUAL, 
            save_videos=SAVE_VIDEOS
        )
        
        # 결과 시각화
        tester.visualize_results(results)
        
        # 에피소드 재생 시각화 생성
        if CREATE_PLAYBACK and tester.episode_frames:
            tester.create_episode_playback()
        
        # 상세 보고서 저장
        tester.save_detailed_report(results)
        
        # 결과 요약 출력
        tester.print_summary(results)
        
        # 생성된 파일 목록
        print(f"\n 생성된 파일들:")
        generated_files = []
        if os.path.exists("road_test_results.png"):
            generated_files.append("road_test_results.png")
        if os.path.exists("road_test_report.json"):
            generated_files.append("road_test_report.json")
        if os.path.exists("episode_playback.gif"):
            generated_files.append("episode_playback.gif")
        
        # 비디오 파일들 찾기
        for file in os.listdir('.'):
            if file.endswith('.mp4') or file.endswith('.gif'):
                if file not in generated_files:
                    generated_files.append(file)
        
        for file in generated_files:
            print(f"   • {file}")
        
    except Exception as e:
        print(f" 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 리소스 정리
        tester.cleanup()
        print("\n 테스트 완료")

if __name__ == "__main__":
    main() 