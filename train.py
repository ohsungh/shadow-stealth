import os
import glob
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

# 로컬 파일에서 커스텀 환경 임포트
from environment import ShadowStealthEnv

# --- 설정 ---
LOG_DIR = "tensorboard_logs"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_FREQ = 10000
TOTAL_TIMESTEPS = 1_000_000
FINAL_MODEL_PATH = "final_model.zip"

def find_latest_checkpoint(directory):
    """지정된 디렉토리에서 가장 최근의 체크포인트 파일을 찾습니다."""
    list_of_files = glob.glob(os.path.join(directory, '*.zip'))
    if not list_of_files:
        return None
    # 파일 이름에서 스텝 수를 추출하여 가장 큰 값을 가진 파일을 찾음
    latest_file = max(list_of_files, key=lambda f: int(os.path.basename(f).split('_')[-2]))
    return latest_file

def train_agent():
    """강화학습 에이전트를 학습시키는 메인 함수"""
    # 로그 및 체크포인트 디렉토리 생성
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 1. 환경 임포트 및 생성
    # 학습 시에는 렌더링이 필요 없으므로 'rgb_array' 모드를 사용합니다.
    # make_vec_env를 사용하여 여러 환경을 병렬로 실행할 수 있습니다. (성능 향상)
    env = make_vec_env(ShadowStealthEnv, n_envs=4, env_kwargs={'render_mode': 'rgb_array'})

    # 2. TensorBoard 로깅 설정은 모델 생성 시 전달됩니다.

    # 3. 주기적 체크포인트 저장 설정
    checkpoint_callback = CheckpointCallback(
        save_freq=max(CHECKPOINT_FREQ // env.num_envs, 1),
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_shadow_stealth",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # 4. 체크포인트에서 학습 재개 로직
    latest_checkpoint = find_latest_checkpoint(CHECKPOINT_DIR)
    if latest_checkpoint:
        print(f"--- 체크포인트에서 학습을 재개합니다: {latest_checkpoint} ---")
        model = PPO.load(
            latest_checkpoint, 
            env=env, 
            tensorboard_log=LOG_DIR,
            # custom_objects를 통해 이전 학습 상태를 완벽하게 복원할 수 있습니다.
        )
    else:
        print("--- 새로운 학습 세션을 시작합니다. ---")
        # PPO 모델 초기화
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            tensorboard_log=LOG_DIR,
            gamma=0.99,
            gae_lambda=0.95,
            n_steps=2048,
            ent_coef=0.01,
            learning_rate=3e-4,
            clip_range=0.2
        )

    # 5. 학습 루프 실행
    print(f"--- 총 {TOTAL_TIMESTEPS} 타임스텝 동안 학습을 시작합니다. ---")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        tb_log_name="PPO_ShadowStealth",
        reset_num_timesteps=False  # 체크포인트에서 이어할 때 타임스텝을 리셋하지 않음
    )

    # 6. 최종 모델 저장
    print(f"--- 학습 완료. 최종 모델을 {FINAL_MODEL_PATH}에 저장합니다. ---")
    model.save(FINAL_MODEL_PATH)

    # 환경 종료
    env.close()

if __name__ == '__main__':
    train_agent()