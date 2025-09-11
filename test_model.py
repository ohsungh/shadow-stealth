import time
import pygame
from stable_baselines3 import PPO
from environment import ShadowStealthEnv

# --- 설정 ---
MODEL_PATH = "final_model.zip"

def watch_agent_play():
    """학습된 AI 에이전트가 게임하는 것을 보여줍니다."""

    # 1. 환경 생성 (사람이 볼 수 있도록 'human' 모드로 설정)
    env = ShadowStealthEnv(render_mode='human')

    # 2. 학습된 모델 불러오기
    try:
        model = PPO.load(MODEL_PATH, env=env)
        print(f"--- 훈련된 모델({MODEL_PATH})을 불러왔습니다. ---")
    except FileNotFoundError:
        print(f"에러: 모델 파일({MODEL_PATH})을 찾을 수 없습니다.")
        print("먼저 train.py를 실행하여 모델을 훈련시켜 주세요.")
        env.close()
        return

    # 3. AI 플레이 시작
    obs, info = env.reset()
    print("--- AI가 게임을 시작합니다. Pygame 창을 확인하세요. ---")
    
    while True:
        # Pygame 창의 이벤트를 처리하여 '닫기' 버튼이 작동하도록 함
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("--- 사용자가 창을 닫았습니다. 테스트를 종료합니다. ---")
                env.close()
                return

        # AI가 다음 행동을 결정
        action, _states = model.predict(obs, deterministic=True)
        
        # 결정한 행동을 환경에서 실행
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 환경을 화면에 렌더링
        env.render()

        # 에피소드가 끝나면 (성공 또는 실패) 재시작
        if terminated or truncated:
            print("에피소드가 종료되었습니다. 2초 후 다시 시작합니다...")
            time.sleep(2)
            obs, info = env.reset()

if __name__ == '__main__':
    watch_agent_play()