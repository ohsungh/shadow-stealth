import torch
import onnxruntime as ort
import numpy as np
from stable_baselines3 import PPO
from environment import ShadowStealthEnv
import os

# --- 설정 ---
# 체크포인트 폴더 안에 있는, 변환할 모델의 경로
model_path = "final_model.zip"

# 최종적으로 생성될 ONNX 모델의 이름
ONNX_MODEL_PATH = "model.onnx"

def convert_model_to_onnx():
    """Stable Baselines3 모델을 ONNX 형식으로 변환합니다."""

    print(f"--- 모델 변환을 시작합니다: {model_path} -> {ONNX_MODEL_PATH} ---")

    # 1. 환경을 생성하여 observation_space 정보를 가져옵니다.
    # 이 정보는 모델의 입력 크기를 알아내는 데 필요합니다.
    env = ShadowStealthEnv()
    obs_space = env.observation_space
    env.close()

    # 2. SB3 PPO 모델을 불러옵니다.
    # CPU에서 실행하도록 device='cpu'를 명시합니다.
    try:
        model = PPO.load(model_path, device='cpu')
        print("SB3 모델을 성공적으로 불러왔습니다.")
    except FileNotFoundError:
        print(f"에러: 모델 파일({model_path})을 찾을 수 없습니다.")
        print("Colab의 checkpoints 폴더에 파일이 있는지 확인해주세요.")
        return

    # 3. ONNX로 내보내기 위한 더미 입력(dummy input)을 생성합니다.
    # 모델이 기대하는 입력과 동일한 모양과 타입이어야 합니다.
    dummy_input = torch.randn(1, *obs_space.shape, device='cpu')
    print(f"모델의 입력 형태(shape): {obs_space.shape}")

    # 4. 모델을 ONNX 형식으로 변환합니다.
    try:
        torch.onnx.export(
            model.policy,
            dummy_input,
            ONNX_MODEL_PATH,
            opset_version=11, # 웹 환경과 호환성을 위해 12 버전을 사용
            input_names=['observation'],
            output_names=['action'],
            dynamic_axes={
                'observation': {0: 'batch_size'},
                'action': {0: 'batch_size'}
            }
        )
        print(f"--- ONNX 모델 변환 성공! 파일이 {ONNX_MODEL_PATH} 로 저장되었습니다. ---")
    except Exception as e:
        print(f"ONNX 변환 중 에러가 발생했습니다: {e}")
        return

    # 5. 변환된 ONNX 모델을 검증합니다.
    print("--- 변환된 모델 검증 시작... ---")
    try:
        ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
        dummy_numpy_input = dummy_input.cpu().numpy()
        
        # ONNX 런타임으로 추론 실행
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_numpy_input}
        ort_outs = ort_session.run(None, ort_inputs)
        
        print("ONNX 런타임으로 추론 실행 성공!")
        print(f"모델 출력 (action): {ort_outs[0]}")
        print("--- 모델 검증 완료. ONNX 파일이 정상입니다. ---")
    except Exception as e:
        print(f"ONNX 모델 검증 중 에러가 발생했습니다: {e}")

if __name__ == '__main__':
    convert_model_to_onnx()