import os
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import glob

# --- 1. 설정 변수 정의 ---
# JSON 파일들이 저장된 폴더 경로를 지정하세요.
# 프로젝트 루트의 'Data' 폴더를 참조합니다.
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = os.path.join(script_dir, "..", "..", "Data")

# 모든 시퀀스의 길이를 통일하기 위한 최대 프레임 수
# (예: 5초 * 60fps = 300)
MAX_SEQUENCE_LENGTH = 300

# 동작(동사) 라벨을 숫자로 매핑
LABEL_MAP = {"Pick": 0, "Hold": 1, "Place": 2}

# --- 2. 파일명에서 라벨 추출 함수 ---
def extract_label_from_filename(filename):
    """
    파일명에서 동작 라벨을 추출합니다.
    지원 형식:
    - Hold_001.json -> Hold
    - Gaze_Hold_010.json -> Hold
    - Pick_002.json -> Pick
    - Place_003.json -> Place
    """
    # Gaze_ 접두사가 있는 경우 제거
    if filename.startswith("Gaze_"):
        filename = filename[5:]  # "Gaze_" 제거
    
    # 첫 번째 언더스코어 이전의 부분을 라벨로 사용
    label_str = filename.split('_')[0]
    return label_str

# --- 3. 데이터 평탄화 함수 ---
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
    # 디버깅: 경로 정보 출력
    current_dir = os.getcwd()
    data_dir = os.path.abspath(DATA_DIRECTORY)
    print(f"📁 현재 작업 디렉토리: {current_dir}")
    print(f"📁 데이터 디렉토리: {data_dir}")
    print(f"📁 상대 경로: {DATA_DIRECTORY}")
    
    # 기존 JSON 파일들과 H2O JSON 파일들을 모두 찾기
    json_files = glob.glob(os.path.join(DATA_DIRECTORY, "*.json"))
    h2o_files = glob.glob(os.path.join(DATA_DIRECTORY, "*_h2o_*.json"))
    
    # 중복 제거 (H2O 파일이 기존 파일과 겹칠 수 있음)
    all_files = []
    existing_names = set()
    
    # 먼저 기존 파일들 추가
    for file_path in json_files:
        filename = os.path.basename(file_path)
        if not filename.startswith(("Hold_h2o_", "Pick_h2o_", "Place_h2o_")):
            all_files.append(file_path)
            existing_names.add(filename)
    
    # H2O 파일들 추가
    for file_path in h2o_files:
        filename = os.path.basename(file_path)
        if filename not in existing_names:
            all_files.append(file_path)
    
    # 디버깅: 찾은 파일들 출력
    print(f"🔍 검색 패턴: {os.path.join(DATA_DIRECTORY, '*.json')} + {os.path.join(DATA_DIRECTORY, '*_h2o_*.json')}")
    print(f"📄 찾은 JSON 파일들:")
    for i, file_path in enumerate(all_files):
        print(f"   {i+1:2d}. {file_path}")
    
    # H2O 파일 개수 별도 출력
    print(f"\n🔍 H2O 데이터셋 파일들:")
    for i, file_path in enumerate(h2o_files):
        print(f"   {i+1:2d}. {os.path.basename(file_path)}")
    
    all_sequences = []
    all_labels = []
    num_features = -1

    print(f"\n총 {len(all_files)}개의 JSON 파일을 찾았습니다. 전처리를 시작합니다...")
    print(f"   - 기존 파일: {len(json_files)}개")
    print(f"   - H2O 파일: {len(h2o_files)}개")
    
    if len(all_files) == 0:
        print("❌ JSON 파일을 찾을 수 없습니다!")
        print("   - 데이터 디렉토리가 존재하는지 확인하세요")
        print("   - 파일 확장자가 .json인지 확인하세요")
        print("   - 경로 설정을 확인하세요")
        return

    # 디버깅: 파일별 라벨 정보 출력
    print("\n🔍 파일별 라벨 추출 정보:")
    for file_path in all_files[:20]:  # 처음 20개만 출력
        filename = os.path.basename(file_path)
        label_str = extract_label_from_filename(filename)
        if label_str in LABEL_MAP:
            print(f"   {filename} -> {label_str} -> {LABEL_MAP[label_str]}")
        else:
            print(f"   {filename} -> {label_str} -> ❌ 알 수 없는 라벨")
    
    if len(all_files) > 20:
        print(f"   ... 및 {len(all_files) - 20}개 더")
    
    # 라벨별 파일 개수 확인
    label_counts = {}
    for file_path in all_files:
        filename = os.path.basename(file_path)
        label_str = extract_label_from_filename(filename)
        if label_str in LABEL_MAP:
            label_num = LABEL_MAP[label_str]
            label_counts[label_num] = label_counts.get(label_num, 0) + 1
        else:
            print(f"⚠️ 알 수 없는 라벨: {filename} -> {label_str}")
    
    print(f"\n📊 라벨별 파일 개수:")
    for label_num, count in sorted(label_counts.items()):
        label_name = [k for k, v in LABEL_MAP.items() if v == label_num][0]
        print(f"   {label_name} (클래스 {label_num}): {count}개")

    for file_path in all_files:
        # 파일명에서 라벨 추출 (Gaze_ 접두사 지원)
        label_str = extract_label_from_filename(os.path.basename(file_path))
        if label_str not in LABEL_MAP:
            print(f"   ⚠️ 알 수 없는 라벨 '{label_str}' 파일 건너뛰기: {os.path.basename(file_path)}")
            continue
        
        print(f"   📁 처리 중: {os.path.basename(file_path)} -> {label_str}")
        all_labels.append(LABEL_MAP[label_str])
        
        try:
            # UTF-8 BOM을 자동으로 처리하기 위해 utf-8-sig 사용
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"   ❌ JSON 파싱 오류: {os.path.basename(file_path)} - {e}")
            # 손상된 파일은 건너뛰기
            all_labels.pop()  # 추가했던 라벨 제거
            continue
        except Exception as e:
            print(f"   ❌ 파일 읽기 오류: {os.path.basename(file_path)} - {e}")
            # 오류가 있는 파일은 건너뛰기
            all_labels.pop()  # 추가했던 라벨 제거
            continue
        
        # frames 데이터가 있는지 확인
        if "frames" not in data or not data["frames"]:
            print(f"   ⚠️ frames 데이터 없음: {os.path.basename(file_path)}")
            all_labels.pop()  # 추가했던 라벨 제거
            continue
        
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
    output_filename = "../preprocessed_data.npz"
    np.savez_compressed(output_filename, X=X_data, y=y_data)

    print(f"\n전처리 완료! '{output_filename}' 파일에 데이터가 저장되었습니다.")
    print(f"저장된 데이터 형태 (X): {X_data.shape}")
    print(f"저장된 라벨 형태 (y): {y_data.shape}")

# 스크립트 실행
if __name__ == "__main__":
    preprocess_data()