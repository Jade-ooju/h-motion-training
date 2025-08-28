import os
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import glob

# --- 1. 설정 변수 정의 ---
# JSON 파일들이 저장된 폴더 경로를 지정하세요.
# 예: "C:/MyProject/UnityData"
# 현재 스크립트와 같은 폴더에 'data' 폴더가 있다고 가정합니다.
DATA_DIRECTORY = "Data"

# 모든 시퀀스의 길이를 통일하기 위한 최대 프레임 수
# (예: 5초 * 60fps = 300)
MAX_SEQUENCE_LENGTH = 300

# 동작(동사) 라벨을 숫자로 매핑
LABEL_MAP = {"Pick": 0, "Hold": 1, "Place": 2}

# --- 2. 데이터 평탄화 함수 ---
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
        
        # 디버깅: 실제 관절 이름들 출력
        if len(flat_data) == 0:  # 첫 번째 프레임에서만 출력
            actual_joint_names = [joint['jointName'] for joint in joints_data]
            print(f"   실제 관절 이름들: {actual_joint_names}")
            print(f"   예상 관절 이름들: {joint_order}")
        
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
    
    # 디버깅: 벡터 길이 확인
    expected_length = 26 * 7 + 3  # 26 joints * 7 values + 3 gaze
    if len(flat_data) != expected_length:
        print(f"   ⚠️ 벡터 길이 불일치: 예상 {expected_length}, 실제 {len(flat_data)}")
    
    return flat_data

# --- 3. 메인 전처리 로직 ---
def preprocess_data():
    """데이터 폴더에서 모든 JSON을 읽어 전처리를 수행합니다."""
    json_files = glob.glob(os.path.join(DATA_DIRECTORY, "*.json"))
    
    all_sequences = []
    all_labels = []
    num_features = -1

    print(f"총 {len(json_files)}개의 JSON 파일을 찾았습니다. 전처리를 시작합니다...")

    for file_path in json_files:
        # 파일명에서 라벨 추출 (예: 'Pick_001.json' -> 'Pick')
        label_str = os.path.basename(file_path).split('_')[0]
        if label_str not in LABEL_MAP:
            continue
        
        all_labels.append(LABEL_MAP[label_str])
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        sequence = [flatten_frame(frame) for frame in data["frames"]]
        
        # 디버깅: 시퀀스 길이 확인
        if len(sequence) > 0:
            first_frame_length = len(sequence[0])
            print(f"   첫 번째 프레임 특징 수: {first_frame_length}")
            
            # 모든 프레임의 길이가 일치하는지 확인
            for i, frame_data in enumerate(sequence):
                if len(frame_data) != first_frame_length:
                    print(f"   ⚠️ 프레임 {i} 길이 불일치: {len(frame_data)} (예상: {first_frame_length})")
                    # 길이가 다른 프레임은 0으로 채워서 맞춤
                    if len(frame_data) < first_frame_length:
                        frame_data.extend([0] * (first_frame_length - len(frame_data)))
                    else:
                        frame_data = frame_data[:first_frame_length]
                    sequence[i] = frame_data
        
        all_sequences.append(sequence)

        # 첫 번째 파일 처리 후 특징(feature) 개수 저장
        if num_features == -1:
            num_features = len(sequence[0])
            print(f"프레임당 특징 개수: {num_features}")

    # --- 4. 패딩 (Padding) ---
    # 모든 시퀀스의 길이를 MAX_SEQUENCE_LENGTH로 통일
    padded_sequences = np.zeros((len(all_sequences), MAX_SEQUENCE_LENGTH, num_features), dtype=np.float32)

    for i, seq in enumerate(all_sequences):
        seq_len = min(len(seq), MAX_SEQUENCE_LENGTH)
        padded_sequences[i, :seq_len, :] = np.array(seq)[:seq_len]

    print(f"패딩 완료. 모든 시퀀스가 (길이: {MAX_SEQUENCE_LENGTH}, 특징: {num_features}) 형태로 통일되었습니다.")

    # --- 5. 정규화 (Normalization) ---
    # 3D 배열을 2D로 펼쳐서 스케일러 적용
    num_samples, seq_len, num_feats = padded_sequences.shape
    reshaped_sequences = padded_sequences.reshape(-1, num_feats)
    
    scaler = MinMaxScaler()
    normalized_sequences_2d = scaler.fit_transform(reshaped_sequences)
    
    # 다시 원래의 3D 형태로 복원
    X_data = normalized_sequences_2d.reshape(num_samples, seq_len, num_feats)
    y_data = np.array(all_labels)

    print("정규화 완료. 모든 특징 값이 0과 1 사이로 조정되었습니다.")

    # --- 6. 데이터 저장 ---
    output_filename = "preprocessed_data.npz"
    np.savez_compressed(output_filename, X=X_data, y=y_data)

    print(f"\n전처리 완료! '{output_filename}' 파일에 데이터가 저장되었습니다.")
    print(f"저장된 데이터 형태 (X): {X_data.shape}")
    print(f"저장된 라벨 형태 (y): {y_data.shape}")

# 스크립트 실행
if __name__ == "__main__":
    preprocess_data()