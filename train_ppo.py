#!/usr/bin/env python3
"""
PPO를 사용한 Mixed Road 환경 훈련 (M1 Mac 최적화)
==============================================

이 스크립트는 Stable-Baselines3의 PPO 알고리즘을 사용하여
custom-mixed-road 환경에서 에이전트를 훈련합니다.

M1 Mac 최적화 특징:
- 단일 환경 훈련 (병렬 처리 제거)
- MPS(Metal Performance Shaders) GPU 지원
- 간소화된 설정으로 빠른 시작
- 실시간 훈련 모니터링
"""

import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import torch
import os
import time
from typing import Dict, Any

class TrainingConfig:
    """M1 Mac용 간소화된 훈련 설정"""
    
    def __init__(self):
        # 환경 설정
        self.env_id = "custom-mixed-road-v0"
        self.max_episode_steps = 1000
        
        # PPO 하이퍼파라미터 (M1 Mac 최적화)
        self.total_timesteps = 100_000  # 로컬 테스트용으로 줄임
        self.learning_rate = 3e-4
        self.n_steps = 1024  # 단일 환경용으로 줄임
        self.batch_size = 64
        self.n_epochs = 4    # 빠른 훈련을 위해 줄임
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_range = 0.2
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        
        # 환경별 설정
        self.env_config = {
            "observation": {"type": "Kinematics"},
            "action": {"type": "DiscreteMetaAction"},
            "vehicles_count": 3,   # 차량 수 대폭 감소 (10 -> 3)
            "controlled_vehicles": 1,
            "duration": 100,       # 에피소드 길이 증가 (60 -> 100)
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "normalize_reward": False,  # 정규화 비활성화로 보상 확인
            "collision_reward": -50,    # 충돌 패널티 증가
            "offroad_terminal": False,  # 도로 이탈시 즉시 종료 방지
            "roundabout_exit_target": "north",
            "success_reward": 100.0,    # 성공 보상 증가
            "completion_distance": 30,
        }
        
        # 로깅 및 저장
        self.log_dir = "./logs/"
        self.model_dir = "./models/"
        self.tensorboard_log = "./tensorboard/"
        self.eval_freq = 5000   # 평가 주기
        self.save_freq = 10000  # 모델 저장 주기
        
        # M1 Mac GPU 지원
        self.device = self._get_device()
        
        # 디렉토리 생성
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.tensorboard_log, exist_ok=True)
    
    def _get_device(self):
        """M1 Mac에 최적화된 디바이스 선택"""
        if torch.backends.mps.is_available():
            print("🚀 MPS (Metal Performance Shaders) 사용 가능!")
            return "mps"
        elif torch.cuda.is_available():
            print("🎮 CUDA GPU 사용 가능!")
            return "cuda"
        else:
            print("💻 CPU 모드로 실행")
            return "cpu"

class SimpleProgressCallback(BaseCallback):
    """간단한 진행 상황 콜백"""
    
    def __init__(self, check_freq: int = 1000):
        super().__init__()
        self.check_freq = check_freq
        self.start_time = None
        
    def _on_training_start(self) -> None:
        self.start_time = time.time()
        print("🎯 훈련 시작!")
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            elapsed_time = time.time() - self.start_time
            progress = self.n_calls / self.locals.get('total_timesteps', 1)
            
            print(f"📊 진행률: {progress:.1%} ({self.n_calls:,} 스텝)")
            print(f"   시간: {elapsed_time/60:.1f}분")
            
            # 최근 에피소드 정보
            if 'infos' in self.locals and self.locals['infos']:
                info = self.locals['infos'][0]
                if 'episode' in info:
                    ep_info = info['episode']
                    print(f"   최근 에피소드: 보상={ep_info['r']:.2f}, 길이={ep_info['l']}")
        
        return True

def create_single_env(config: TrainingConfig):
    """단일 환경 생성"""
    env = gym.make(config.env_id)
    env.unwrapped.configure(config.env_config)
    env = Monitor(env, config.log_dir + "training_env")
    return env

def evaluate_model_simple(model, env, n_episodes: int = 5):
    """간단한 모델 평가"""
    print(f"\n🧪 모델 평가 중... ({n_episodes}개 에피소드)")
    
    episode_rewards = []
    success_count = 0
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        
        # 성공 여부 확인
        if episode_reward > 30:
            success_count += 1
        
        print(f"   Episode {episode+1}: 보상={episode_reward:.2f}")
    
    avg_reward = np.mean(episode_rewards)
    success_rate = success_count / n_episodes
    
    print(f"📊 평가 결과: 평균 보상={avg_reward:.2f}, 성공률={success_rate:.1%}")
    
    return avg_reward, success_rate

def plot_simple_progress(log_dir: str, save_path: str):
    """간단한 훈련 진행 상황 시각화"""
    try:
        import pandas as pd
        
        # 로그 파일 찾기
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.monitor.csv')]
        if not log_files:
            print("❌ 로그 파일을 찾을 수 없습니다.")
            return
        
        # 데이터 읽기
        df = pd.read_csv(os.path.join(log_dir, log_files[0]), skiprows=1)
        
        # 시각화
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('PPO 훈련 진행 상황', fontsize=14)
        
        # 에피소드 보상
        ax1.plot(df['r'], alpha=0.7)
        if len(df) > 10:
            # 이동평균
            window = min(20, len(df) // 5)
            moving_avg = df['r'].rolling(window=window).mean()
            ax1.plot(moving_avg, color='red', linewidth=2, label=f'이동평균({window})')
            ax1.legend()
        
        ax1.set_title('에피소드 보상')
        ax1.set_xlabel('에피소드')
        ax1.set_ylabel('보상')
        ax1.grid(True, alpha=0.3)
        
        # 에피소드 길이
        ax2.plot(df['l'], alpha=0.7, color='green')
        ax2.set_title('에피소드 길이')
        ax2.set_xlabel('에피소드')
        ax2.set_ylabel('스텝 수')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 훈련 진행 상황 저장: {save_path}")
        plt.close()
        
    except Exception as e:
        print(f"❌ 시각화 실패: {e}")

def main():
    """M1 Mac용 간소화된 메인 훈련 함수"""
    print("🚗 Mixed Road PPO 훈련 (M1 Mac 최적화)")
    print("=" * 50)
    
    # 설정 로드
    config = TrainingConfig()
    
    print(f"🔧 훈련 설정:")
    print(f"   • 총 훈련 스텝: {config.total_timesteps:,}")
    print(f"   • 디바이스: {config.device}")
    print(f"   • 차량 수: {config.env_config['vehicles_count']}")
    print(f"   • 에피소드 길이: {config.env_config['duration']}")
    
    try:
        # 환경 생성
        print("\n🔧 환경 설정 중...")
        train_env = create_single_env(config)
        eval_env = create_single_env(config)
        print("✅ 환경 생성 완료")
        
        # PPO 모델 생성
        print("\n🧠 PPO 모델 생성 중...")
        
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            tensorboard_log=config.tensorboard_log,
            device=config.device,
            verbose=1
        )
        
        print(f"✅ PPO 모델 생성 완료")
        
        # 콜백 설정
        progress_callback = SimpleProgressCallback(check_freq=1000)
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=config.model_dir,
            log_path=config.log_dir,
            eval_freq=config.eval_freq,
            deterministic=True,
            render=False,
            verbose=1,
            n_eval_episodes=3  # 빠른 평가를 위해 줄임
        )
        
        # 훈련 시작
        print(f"\n🎯 훈련 시작!")
        print("   TensorBoard 로그를 보려면: tensorboard --logdir=./tensorboard/")
        print("   Ctrl+C로 언제든 중단할 수 있습니다.")
        
        start_time = time.time()
        
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=[progress_callback, eval_callback],
            tb_log_name="PPO_MixedRoad_M1"
        )
        
        training_time = time.time() - start_time
        print(f"\n🏁 훈련 완료! (소요 시간: {training_time/60:.1f}분)")
        
        # 최종 모델 저장
        final_model_path = os.path.join(config.model_dir, "ppo_mixed_road_m1.zip")
        model.save(final_model_path)
        print(f"💾 최종 모델 저장: {final_model_path}")
        
        # 최종 평가
        print("\n🏆 최종 모델 평가")
        final_avg_reward, final_success_rate = evaluate_model_simple(
            model, eval_env, n_episodes=10
        )
        
        # 베스트 모델 평가 (있는 경우)
        best_model_path = os.path.join(config.model_dir, "best_model.zip")
        if os.path.exists(best_model_path):
            print("\n🥇 베스트 모델 평가")
            best_model = PPO.load(best_model_path)
            best_avg_reward, best_success_rate = evaluate_model_simple(
                best_model, eval_env, n_episodes=10
            )
            
            print(f"\n📈 결과 비교:")
            print(f"   최종 모델: 보상={final_avg_reward:.2f}, 성공률={final_success_rate:.1%}")
            print(f"   베스트 모델: 보상={best_avg_reward:.2f}, 성공률={best_success_rate:.1%}")
        
        # 훈련 진행 상황 시각화
        plot_path = os.path.join(config.model_dir, "training_progress.png")
        plot_simple_progress(config.log_dir, plot_path)
        
        print(f"\n🎉 훈련 완료!")
        print(f"   • 모델 저장 위치: {config.model_dir}")
        print(f"   • 로그 위치: {config.log_dir}")
        print(f"   • 다음 명령어로 테스트: python test_trained_model.py")
        
    except KeyboardInterrupt:
        print("\n⚠️  사용자에 의해 훈련이 중단되었습니다.")
        print("   부분적으로 훈련된 모델이 저장되었을 수 있습니다.")
        
    except Exception as e:
        print(f"\n❌ 훈련 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 환경 정리
        try:
            train_env.close()
            eval_env.close()
        except:
            pass

if __name__ == "__main__":
    main() 