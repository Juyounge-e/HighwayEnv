#!/usr/bin/env python3
"""
훈련된 PPO 모델 테스트 및 시각화
===============================

이 스크립트는 훈련된 PPO 모델을 로드하고 테스트합니다.
시각적 확인과 성능 분석을 제공합니다.
"""

import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter
from stable_baselines3 import PPO
import os
import cv2
import json
from typing import Dict, List, Optional

class ModelTester:
    """훈련된 모델 테스트 클래스"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.env = None
        self.test_results = []
        
    def load_model(self):
        """모델 로드"""
        print(f"🧠 모델 로드 중: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
        
        self.model = PPO.load(self.model_path)
        print("✅ 모델 로드 완료")
        
    def setup_environment(self):
        """테스트 환경 설정"""
        print("🔧 테스트 환경 설정 중...")
        
        self.env = gym.make("custom-mixed-road-v0", render_mode="rgb_array")
        
        # 테스트용 환경 설정
        self.env.unwrapped.configure({
            "observation": {"type": "Kinematics"},
            "action": {"type": "DiscreteMetaAction"},
            "vehicles_count": 15,  # 훈련과 동일한 설정
            "controlled_vehicles": 1,
            "duration": 100,
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "normalize_reward": True,
            "collision_reward": -5,
            "offroad_terminal": True,
            "roundabout_exit_target": "north",
            "success_reward": 50.0,
            "completion_distance": 30,
        })
        
        print("✅ 테스트 환경 설정 완료")
    
    def run_single_episode(self, episode_num: int = 1, render: bool = True, 
                          save_video: bool = False) -> Dict:
        """단일 에피소드 실행"""
        print(f"\n🎮 에피소드 {episode_num} 실행 중...")
        
        obs, info = self.env.reset()
        
        episode_data = {
            'episode': episode_num,
            'trajectory': [],
            'frames': [],
            'total_reward': 0,
            'steps': 0,
            'success': False,
            'crash': False,
            'segments_visited': []
        }
        
        # 시각화 설정
        if render:
            plt.ion()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            ax1.set_title(f'Episode {episode_num} - 실시간 주행')
            ax2.set_title('주행 궤적')
        
        done = False
        step = 0
        
        while not done and step < 1000:  # 최대 1000 스텝
            # 모델 예측
            action, _states = self.model.predict(obs, deterministic=True)
            
            # 환경 스텝
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # 프레임 캡처
            if save_video or render:
                frame = self.env.render()
                if frame is not None:
                    episode_data['frames'].append(frame)
            
            # 차량 상태 기록
            ego = self.env.unwrapped.vehicle
            if ego and hasattr(ego, 'position') and ego.position is not None:
                position = ego.position.copy()
                speed = ego.speed
                crashed = ego.crashed
                on_road = ego.on_road
                
                # 현재 구간 정보
                current_segment = "unknown"
                if hasattr(ego, 'lane_index') and ego.lane_index:
                    lane_key = ego.lane_index[:2]
                    segment_labels = getattr(self.env.unwrapped, 'segment_labels', {})
                    current_segment = segment_labels.get(lane_key, f"{lane_key[0]}->{lane_key[1]}")
                
                episode_data['trajectory'].append({
                    'step': step,
                    'position': position,
                    'speed': speed,
                    'crashed': crashed,
                    'on_road': on_road,
                    'reward': reward,
                    'action': action,
                    'segment': current_segment
                })
                
                episode_data['segments_visited'].append(current_segment)
                episode_data['total_reward'] += reward
                
                # 실시간 시각화 업데이트
                if render and step % 10 == 0:
                    # 현재 프레임
                    ax1.clear()
                    if episode_data['frames']:
                        ax1.imshow(episode_data['frames'][-1])
                        ax1.set_title(f'Step {step} - Speed: {speed:.1f} - Reward: {reward:.2f}')
                        ax1.axis('off')
                    
                    # 궤적 업데이트
                    ax2.clear()
                    positions = [t['position'] for t in episode_data['trajectory']]
                    x_coords = [pos[0] for pos in positions]
                    y_coords = [pos[1] for pos in positions]
                    
                    ax2.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.7, label='궤적')
                    ax2.scatter(x_coords[-1], y_coords[-1], color='red', s=100, 
                              label='현재 위치', zorder=5)
                    
                    # 회전교차로 표시
                    if hasattr(self.env.unwrapped, 'roundabout_center'):
                        center = self.env.unwrapped.roundabout_center
                        radius = self.env.unwrapped.roundabout_radius
                        circle = Circle(center, radius, fill=False, color='orange', linewidth=2)
                        ax2.add_patch(circle)
                    
                    ax2.set_xlabel('X (m)')
                    ax2.set_ylabel('Y (m)')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    plt.pause(0.05)
                
                # 상태 체크
                if crashed:
                    episode_data['crash'] = True
                    print(f"   ❌ Step {step}에서 충돌 발생!")
                    break
            
            step += 1
        
        # 에피소드 완료 처리
        episode_data['steps'] = step
        
        # 성공 여부 판단
        if episode_data['total_reward'] > 30 and not episode_data['crash']:
            episode_data['success'] = True
        
        # 시각화 정리
        if render:
            plt.ioff()
            plt.close(fig)
        
        # 비디오 저장
        if save_video and episode_data['frames']:
            self._save_video(episode_data['frames'], f"episode_{episode_num}_test.mp4")
        
        # 결과 출력
        status = "✅ 성공" if episode_data['success'] else "❌ 실패"
        crash_status = "충돌" if episode_data['crash'] else "안전"
        unique_segments = len(set(episode_data['segments_visited']))
        
        print(f"   {status} - {step}스텝, 보상={episode_data['total_reward']:.2f}, {crash_status}")
        print(f"   방문 구간: {unique_segments}개")
        
        return episode_data
    
    def _save_video(self, frames: List[np.ndarray], filename: str):
        """비디오 저장"""
        if not frames:
            return
            
        print(f"   🎥 비디오 저장 중: {filename}")
        
        try:
            height, width, layers = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(filename, fourcc, 10.0, (width, height))
            
            for frame in frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video.write(frame_bgr)
            
            video.release()
            print(f"   ✅ 비디오 저장 완료: {filename}")
            
        except Exception as e:
            print(f"   ❌ 비디오 저장 실패: {e}")
    
    def run_multiple_episodes(self, n_episodes: int = 10, save_videos: bool = False) -> Dict:
        """여러 에피소드 실행 및 통계 분석"""
        print(f"\n📊 {n_episodes}개 에피소드 테스트 시작")
        
        all_results = []
        success_count = 0
        crash_count = 0
        total_rewards = []
        episode_lengths = []
        
        for i in range(n_episodes):
            result = self.run_single_episode(
                episode_num=i+1, 
                render=False,  # 여러 에피소드 실행시 렌더링 비활성화
                save_video=save_videos and i < 3  # 처음 3개만 비디오 저장
            )
            
            all_results.append(result)
            total_rewards.append(result['total_reward'])
            episode_lengths.append(result['steps'])
            
            if result['success']:
                success_count += 1
            if result['crash']:
                crash_count += 1
        
        # 통계 계산
        stats = {
            'n_episodes': n_episodes,
            'success_rate': success_count / n_episodes,
            'crash_rate': crash_count / n_episodes,
            'avg_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'avg_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'max_reward': np.max(total_rewards),
            'min_reward': np.min(total_rewards),
            'all_results': all_results
        }
        
        # 결과 출력
        print(f"\n📈 테스트 결과 요약:")
        print(f"   • 성공률: {stats['success_rate']:.1%} ({success_count}/{n_episodes})")
        print(f"   • 충돌률: {stats['crash_rate']:.1%} ({crash_count}/{n_episodes})")
        print(f"   • 평균 보상: {stats['avg_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"   • 평균 길이: {stats['avg_length']:.1f} ± {stats['std_length']:.1f}")
        print(f"   • 보상 범위: {stats['min_reward']:.2f} ~ {stats['max_reward']:.2f}")
        
        return stats
    
    def visualize_performance(self, stats: Dict, save_path: str = "model_performance.png"):
        """성능 시각화"""
        print(f"\n📊 성능 시각화 중... (저장 경로: {save_path})")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('훈련된 모델 성능 분석', fontsize=16, fontweight='bold')
        
        all_results = stats['all_results']
        
        # 1. 에피소드별 보상
        ax1 = axes[0, 0]
        rewards = [r['total_reward'] for r in all_results]
        ax1.plot(rewards, 'b-o', markersize=4)
        ax1.axhline(y=np.mean(rewards), color='r', linestyle='--', label=f'평균: {np.mean(rewards):.2f}')
        ax1.set_title('에피소드별 보상')
        ax1.set_xlabel('에피소드')
        ax1.set_ylabel('총 보상')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 에피소드별 길이
        ax2 = axes[0, 1]
        lengths = [r['steps'] for r in all_results]
        ax2.plot(lengths, 'g-o', markersize=4)
        ax2.axhline(y=np.mean(lengths), color='r', linestyle='--', label=f'평균: {np.mean(lengths):.1f}')
        ax2.set_title('에피소드별 길이')
        ax2.set_xlabel('에피소드')
        ax2.set_ylabel('스텝 수')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 성공/실패/충돌 분포
        ax3 = axes[0, 2]
        success_count = sum(1 for r in all_results if r['success'])
        crash_count = sum(1 for r in all_results if r['crash'])
        incomplete_count = len(all_results) - success_count - crash_count
        
        labels = ['성공', '충돌', '미완료']
        sizes = [success_count, crash_count, incomplete_count]
        colors = ['green', 'red', 'orange']
        
        ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('결과 분포')
        
        # 4. 보상 히스토그램
        ax4 = axes[1, 0]
        ax4.hist(rewards, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax4.axvline(x=np.mean(rewards), color='r', linestyle='--', label=f'평균: {np.mean(rewards):.2f}')
        ax4.set_title('보상 분포')
        ax4.set_xlabel('보상')
        ax4.set_ylabel('빈도')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 궤적 예시 (성공한 에피소드 중 하나)
        ax5 = axes[1, 1]
        success_episodes = [r for r in all_results if r['success']]
        if success_episodes:
            example = success_episodes[0]
            positions = [t['position'] for t in example['trajectory']]
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            ax5.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.7)
            ax5.scatter(x_coords[0], y_coords[0], color='green', s=100, label='시작', zorder=5)
            ax5.scatter(x_coords[-1], y_coords[-1], color='red', s=100, label='종료', zorder=5)
            
            # 회전교차로 표시
            if hasattr(self.env.unwrapped, 'roundabout_center'):
                center = self.env.unwrapped.roundabout_center
                radius = self.env.unwrapped.roundabout_radius
                circle = Circle(center, radius, fill=False, color='orange', linewidth=2)
                ax5.add_patch(circle)
            
            ax5.set_title('성공 궤적 예시')
            ax5.set_xlabel('X (m)')
            ax5.set_ylabel('Y (m)')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, '성공한 에피소드 없음', ha='center', va='center', 
                    transform=ax5.transAxes, fontsize=12)
            ax5.set_title('성공 궤적 (없음)')
        
        # 6. 통계 요약
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        stats_text = f"""
모델 성능 요약

테스트 에피소드: {stats['n_episodes']}개

성과 지표:
• 성공률: {stats['success_rate']:.1%}
• 충돌률: {stats['crash_rate']:.1%}
• 평균 보상: {stats['avg_reward']:.2f}
• 보상 표준편차: {stats['std_reward']:.2f}
• 평균 길이: {stats['avg_length']:.1f}

보상 범위:
• 최고: {stats['max_reward']:.2f}
• 최저: {stats['min_reward']:.2f}

모델 파일:
{os.path.basename(self.model_path)}
"""
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 성능 시각화 완료: {save_path}")
        
        return fig
    
    def save_results(self, stats: Dict, save_path: str = "test_results.json"):
        """결과 저장"""
        print(f"\n💾 테스트 결과 저장 중... (저장 경로: {save_path})")
        
        # JSON 직렬화를 위한 데이터 정리
        save_data = {
            'model_path': self.model_path,
            'test_summary': {
                'n_episodes': stats['n_episodes'],
                'success_rate': stats['success_rate'],
                'crash_rate': stats['crash_rate'],
                'avg_reward': stats['avg_reward'],
                'std_reward': stats['std_reward'],
                'avg_length': stats['avg_length'],
                'std_length': stats['std_length'],
                'max_reward': stats['max_reward'],
                'min_reward': stats['min_reward']
            },
            'episode_details': []
        }
        
        # 각 에피소드 세부 정보 (궤적 제외)
        for result in stats['all_results']:
            episode_detail = {
                'episode': result['episode'],
                'total_reward': result['total_reward'],
                'steps': result['steps'],
                'success': result['success'],
                'crash': result['crash'],
                'segments_visited': len(set(result['segments_visited']))
            }
            save_data['episode_details'].append(episode_detail)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 테스트 결과 저장 완료: {save_path}")
    
    def cleanup(self):
        """리소스 정리"""
        if self.env:
            self.env.close()

def main():
    """메인 테스트 함수"""
    print("🧪 훈련된 PPO 모델 테스트")
    print("=" * 40)
    
    # 모델 경로 설정 (사용자가 수정 가능)
    model_paths = [
        "./models/best_model.zip",
        "./models/ppo_mixed_road_final.zip"
    ]
    
    # 존재하는 모델 찾기
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("❌ 테스트할 모델을 찾을 수 없습니다.")
        print("   다음 경로에 모델이 있는지 확인하세요:")
        for path in model_paths:
            print(f"   • {path}")
        return
    
    print(f"📁 사용할 모델: {model_path}")
    
    # 테스터 초기화
    tester = ModelTester(model_path)
    
    try:
        # 모델 및 환경 설정
        tester.load_model()
        tester.setup_environment()
        
        # 사용자 선택
        print("\n🎯 테스트 옵션:")
        print("1. 단일 에피소드 (시각화 포함)")
        print("2. 다중 에피소드 성능 분석")
        print("3. 둘 다 실행")
        
        choice = input("선택하세요 (1/2/3): ").strip()
        
        if choice in ['1', '3']:
            # 단일 에피소드 테스트
            print("\n" + "="*50)
            episode_result = tester.run_single_episode(
                episode_num=1, 
                render=True, 
                save_video=True
            )
        
        if choice in ['2', '3']:
            # 다중 에피소드 테스트
            print("\n" + "="*50)
            n_episodes = 20  # 테스트할 에피소드 수
            stats = tester.run_multiple_episodes(n_episodes=n_episodes, save_videos=False)
            
            # 결과 시각화 및 저장
            tester.visualize_performance(stats)
            tester.save_results(stats)
        
        print("\n🎉 테스트 완료!")
        
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        tester.cleanup()

if __name__ == "__main__":
    main() 