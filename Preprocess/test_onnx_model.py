import numpy as np
import onnxruntime as ort
import json
import os
from sklearn.preprocessing import MinMaxScaler

# --- 설정 변수 ---
DATA_DIRECTORY = "Data"
MAX_SEQUENCE_LENGTH = 300

# 동작(동사) 라벨을 숫자로 매핑
LABEL_MAP = {"Pick": 0, "Hold": 1, "Place": 2}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# --- 데이터 평탄화 함수 (preprocess_test.py와 동일) ---
def flatten_frame(frame):
    """단일 프레임의 모든 관절/시선 데이터를 1차원 벡터로 변환합니다."""
    flat_data = []
    
    # 손 관절 데이터를 순서대로 추가 (순서 유지가 매우 중요!)
    joint_order = [
        "Wrist", "ForearmWrist", "Palm", "ThumbMetacarpal", "ThumbProximal", "ThumbDistal", "ThumbTip",
        "IndexMetacarpal", "IndexProximal", "IndexIntermediate", "IndexDistal", "IndexTip",
        "MiddleMetacarpal", "MiddleProximal", "MiddleIntermediate", "MiddleDistal", "MiddleTip",
        "RingMetacarpal", "RingProximal", "RingIntermediate", "RingDistal", "RingTip",
        "PinkyMetacarpal", "PinkyProximal", "PinkyIntermediate", "PinkyDistal"  # Little -> Pinky로 수정
    ]
    
    # joints 배열에서 관절 데이터 추출
    if 'joints' in frame and frame['joints']:
        joints_data = frame['joints']
        
        # 관절 이름을 키로 하는 딕셔너리 생성
        joints_dict = {joint['jointName']: joint for joint in joints_data}
        
        # 순서대로 관절 데이터 추출
        for joint_name in joint_order:
            if joint_name in joints_dict:
                joint = joints_dict[joint_name]
                pos = joint.get('position', {})
                rot = joint.get('rotation', {})
                
                # Position (x, y, z)
                flat_data.extend([pos.get('x', 0), pos.get('y', 0), pos.get('z', 0)])
                # Rotation (x, y, z, w)
                flat_data.extend([rot.get('x', 0), rot.get('y', 0), rot.get('z', 0), rot.get('w', 0)])
            else:
                # 관절이 없는 경우 0으로 채움
                flat_data.extend([0] * 7)
    else:
        # joints 데이터가 없는 경우 0으로 채움 (26 joints * 7 values)
        flat_data.extend([0] * 26 * 7)

    # Gaze Target Position (x, y, z) - 현재는 없으므로 0으로 채움
    # 향후 시선 데이터가 추가되면 여기서 처리
    flat_data.extend([0, 0, 0])
    
    return flat_data

# --- 데이터 전처리 함수 ---
def preprocess_single_file(file_path):
    """단일 JSON 파일을 전처리하여 모델 입력 형태로 변환합니다."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 프레임 데이터를 평탄화
    sequence = [flatten_frame(frame) for frame in data["frames"]]
    
    # 패딩 적용
    num_features = len(sequence[0])
    padded_sequence = np.zeros((MAX_SEQUENCE_LENGTH, num_features), dtype=np.float32)
    
    seq_len = min(len(sequence), MAX_SEQUENCE_LENGTH)
    padded_sequence[:seq_len, :] = np.array(sequence)[:seq_len]
    
    # 정규화 (MinMaxScaler 사용)
    # 실제로는 훈련 시 사용한 스케일러를 저장해서 사용해야 하지만,
    # 여기서는 간단히 0-1 정규화를 적용합니다
    min_vals = np.min(padded_sequence, axis=0)
    max_vals = np.max(padded_sequence, axis=0)
    max_vals = np.where(max_vals == min_vals, 1, max_vals - min_vals)  # 0으로 나누기 방지
    
    normalized_sequence = (padded_sequence - min_vals) / max_vals
    
    return normalized_sequence.reshape(1, MAX_SEQUENCE_LENGTH, -1)

# --- ONNX 모델 테스트 함수 ---
def test_onnx_model():
    """ONNX 모델을 로드하고 테스트 데이터로 예측을 수행합니다."""
    
    # ONNX 모델 파일 확인
    onnx_file = "motion_classifier.onnx"
    if not os.path.exists(onnx_file):
        print(f"❌ ONNX 모델 파일 '{onnx_file}'을 찾을 수 없습니다.")
        print("먼저 'onnx_converter_LSTM.py'를 실행하여 모델을 생성하세요.")
        return
    
    print("🔍 ONNX 모델을 로드하고 있습니다...")
    
    try:
        # ONNX 모델 로드
        session = ort.InferenceSession(onnx_file)
        print("✅ ONNX 모델 로드 완료!")
        
        # 모델 정보 출력
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        input_shape = session.get_inputs()[0].shape
        output_shape = session.get_outputs()[0].shape
        
        print(f"📊 모델 입력 형태: {input_name} -> {input_shape}")
        print(f"📊 모델 출력 형태: {output_name} -> {output_shape}")
        
    except Exception as e:
        print(f"❌ ONNX 모델 로드 실패: {e}")
        return
    
    # 데이터 폴더에서 테스트 파일들 찾기
    json_files = glob.glob(os.path.join(DATA_DIRECTORY, "*.json"))
    
    if not json_files:
        print(f"❌ '{DATA_DIRECTORY}' 폴더에서 JSON 파일을 찾을 수 없습니다.")
        return
    
    print(f"\n🧪 총 {len(json_files)}개의 테스트 파일로 예측을 시작합니다...")
    
    # 각 파일별로 예측 수행
    for i, file_path in enumerate(json_files[:5]):  # 처음 5개 파일만 테스트
        try:
            # 파일명에서 실제 라벨 추출
            actual_label = os.path.basename(file_path).split('_')[0]
            if actual_label not in LABEL_MAP:
                continue
            
            actual_label_num = LABEL_MAP[actual_label]
            
            # 데이터 전처리
            input_data = preprocess_single_file(file_path)
            
            # 예측 실행
            result = session.run([output_name], {input_name: input_data})
            prediction_probs = result[0][0]  # 첫 번째 배치의 결과
            
            # 예측 결과 분석
            predicted_label_num = np.argmax(prediction_probs)
            confidence = prediction_probs[predicted_label_num]
            
            predicted_label = REVERSE_LABEL_MAP[predicted_label_num]
            
            # 결과 출력
            print(f"\n📁 파일: {os.path.basename(file_path)}")
            print(f"   실제 라벨: {actual_label} ({actual_label_num})")
            print(f"   예측 라벨: {predicted_label} ({predicted_label_num})")
            print(f"   신뢰도: {confidence:.3f}")
            print(f"   정확도: {'✅' if predicted_label_num == actual_label_num else '❌'}")
            
            # 모든 클래스별 확률 출력
            print("   클래스별 확률:")
            for j, prob in enumerate(prediction_probs):
                label_name = REVERSE_LABEL_MAP[j]
                marker = "🎯" if j == predicted_label_num else "  "
                print(f"     {marker} {label_name}: {prob:.3f}")
                
        except Exception as e:
            print(f"❌ 파일 '{os.path.basename(file_path)}' 처리 중 오류: {e}")
            continue
    
    print(f"\n🎉 ONNX 모델 테스트 완료!")
    print("이제 Unity에서도 동일한 결과를 얻을 수 있을 것입니다.")

# --- 메인 실행 ---
if __name__ == "__main__":
    import glob
    test_onnx_model()
