# 🚗 Mixed Road PPO 자율주행 강화학습

`highway-env`를 기반으로 고속도로, 합류로, 회전교차로 등 다양한 도로를 연결한 커스텀 환경에서 PPO 알고리즘을 학습시킨 프로젝트입니다.

<br>

## 🌐 프로젝트 개요

- 고속도로 → 합류 구간 → 회전 교차로 → 출구로 이어지는 도로 구성
- 각 구간에 `세그먼트 라벨`을 부여해 구간별 **보상 함수 및 종료 조건 차등 적용**
- PPO (Proximal Policy Optimization) 알고리즘을 이용한 정책 학습
- TensorBoard와 Matplotlib로 학습 진행 시각화
- 성공률, 평균 보상 등 지표 기반 성능 평가

<br>

## 🧱 환경 구성 및 설치

```bash
git clone https://github.com/Juyounge-e/HighwayEnv
cd mixed-road

# 의존 패키지 설치
pip install -r requirements.txt


🏁 실행 방법

# 학습 실행
python train_ppo.py

# 학습된 모델 평가
python test_trained_model.py

# 구성 환경 시각화 
python env_visualize.py


🛣️ 사용자 정의 환경
•	custom_mixed_env.py에서 고속도로, 합류로, 회전교차로를 수작업으로 생성
•	각 도로는 코드로 정의되어 이어 붙였으며, 각 도로에는 세그먼트별 고유 라벨을 부여
•	이 라벨을 기반으로 보상 및 종료 조건을 구간별로 달리 설정


📊 결과 요약

항목	값
에피소드 수	20
성공률	25% (5/20)
평균 보상	173.46
최대 보상	357.67
평균 에피소드 길이	232.5 steps



📁 디렉토리 구조
├── ./highway_env/envs/custom_mixed_env.py  # 사용자 정의 환경
├── train_ppo.py                            # PPO 훈련 스크립트
├── test_trained_model.py                   # 모델 평가 스크립트
├── env_visualize.py                        # 주행 환경 시각화 스크립트
├── models/                                 # 저장된 모델 
├── logs/                                   # 훈련 로그
└── README.md                               # 프로젝트 설명



📚 참고 자료
•	Highway-env GitHub
•	Stable-Baselines3 (PPO)

⸻

✍️ 작성자
**유주영**
산업경영공학전공 
