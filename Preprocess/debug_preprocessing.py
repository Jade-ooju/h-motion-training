import numpy as np
import json
import os
import glob
from sklearn.preprocessing import MinMaxScaler

# --- 설정 변수 ---
DATA_DIRECTORY = "Data"
MAX_SEQUENCE_LENGTH = 300

# 동작(동사) 라벨을 숫자로 매핑
LABEL_MAP = {"Pick": 0, "Hold": 1, "Place": 2}

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
        # joints 데이터가 없는 경우 0으로 채움 (25 joints * 7 values)
        flat_data.extend([0] * 25 * 7)

    # Gaze Target Position (x, y, z) - 현재는 없으므로 0으로 채움
    # 향후 시선 데이터가 추가되면 여기서 처리
    flat_data.extend([0, 0, 0])
    
    return flat_data

# --- 원본 데이터 분석 ---
def analyze_raw_data():
    """원본 JSON 데이터의 구조와 내용을 분석합니다."""
    print("🔍 원본 데이터 분석 시작...")
    
    json_files = glob.glob(os.path.join(DATA_DIRECTORY, "*.json"))
    if not json_files:
        print(f"❌ '{DATA_DIRECTORY}' 폴더에서 JSON 파일을 찾을 수 없습니다.")
        return
    
    print(f"📁 총 {len(json_files)}개의 JSON 파일 발견")
    
    # 클래스별 파일 수 확인
    class_counts = {}
    for file_path in json_files:
        label = os.path.basename(file_path).split('_')[0]
        class_counts[label] = class_counts.get(label, 0) + 1
    
    print("\n📊 클래스별 파일 분포:")
    for label, count in class_counts.items():
        print(f"   {label}: {count}개")
    
    # 첫 번째 파일의 구조 분석
    sample_file = json_files[0]
    print(f"\n🔬 샘플 파일 분석: {os.path.basename(sample_file)}")
    
    with open(sample_file, 'r') as f:
        data = json.load(f)
    
    print(f"   프레임 수: {len(data['frames'])}")
    
    if data['frames']:
        first_frame = data['frames'][0]
        print(f"   첫 번째 프레임 키들: {list(first_frame.keys())}")
        
        # 손 데이터 확인
        for hand in ["LeftHand", "RightHand"]:
            if hand in first_frame:
                hand_data = first_frame[hand]
                if hand_data:
                    print(f"   {hand} 관절 수: {len(hand_data)}")
                    print(f"   {hand} 첫 번째 관절: {list(hand_data.keys())[0] if hand_data else 'None'}")
                else:
                    print(f"   {hand}: 빈 데이터")
            else:
                print(f"   {hand}: 키가 없음")
        
        # 시선 데이터 확인
        if "GazeTargetPosition" in first_frame:
            gaze = first_frame["GazeTargetPosition"]
            print(f"   시선 데이터: {gaze}")
        else:
            print("   시선 데이터: 키가 없음")

# --- 전처리된 데이터 분석 ---
def analyze_preprocessed_data():
    """전처리된 npz 파일의 내용을 분석합니다."""
    print("\n🔍 전처리된 데이터 분석 시작...")
    
    npz_file = "preprocessed_data.npz"
    if not os.path.exists(npz_file):
        print(f"❌ 전처리된 데이터 파일 '{npz_file}'을 찾을 수 없습니다.")
        print("먼저 'preprocess_test.py'를 실행하세요.")
        return
    
    data = np.load(npz_file)
    X = data['X']
    y = data['y']
    
    print(f"📊 데이터 형태:")
    print(f"   X: {X.shape}")
    print(f"   y: {y.shape}")
    
    print(f"\n📊 X 데이터 통계:")
    print(f"   최솟값: {X.min():.6f}")
    print(f"   최댓값: {X.max():.6f}")
    print(f"   평균: {X.mean():.6f}")
    print(f"   표준편차: {X.std():.6f}")
    print(f"   NaN 값: {np.isnan(X).sum()}")
    print(f"   무한대 값: {np.isinf(X).sum()}")
    
    print(f"\n📊 y 데이터 분포:")
    unique, counts = np.unique(y, return_counts=True)
    for label_num, count in zip(unique, counts):
        label_name = list(LABEL_MAP.keys())[list(LABEL_MAP.values()).index(label_num)]
        print(f"   {label_name} ({label_num}): {count}개")
    
    # 특징별 통계
    print(f"\n🔬 특징별 통계 (첫 번째 시퀀스):")
    first_sequence = X[0]  # (300, 185)
    
    # 손 관절 특징 (25 joints * 7 values = 175)
    hand_features = first_sequence[:, :175]
    print(f"   손 관절 특징 (0-174):")
    print(f"     범위: {hand_features.min():.6f} ~ {hand_features.max():.6f}")
    print(f"     평균: {hand_features.mean():.6f}")
    print(f"     표준편차: {hand_features.std():.6f}")
    
    # 시선 특징 (3 values)
    gaze_features = first_sequence[:, 175:]
    print(f"   시선 특징 (175-184):")
    if gaze_features.size > 0:
        print(f"     범위: {gaze_features.min():.6f} ~ {gaze_features.max():.6f}")
        print(f"     평균: {gaze_features.mean():.6f}")
        print(f"     표준편차: {gaze_features.std():.6f}")
    else:
        print(f"     시선 특징이 없습니다.")

# --- 데이터 품질 검증 ---
def validate_data_quality():
    """데이터 품질을 검증합니다."""
    print("\n🔍 데이터 품질 검증 시작...")
    
    # 전처리된 데이터 로드
    npz_file = "preprocessed_data.npz"
    if not os.path.exists(npz_file):
        print("❌ 전처리된 데이터가 없습니다.")
        return
    
    data = np.load(npz_file)
    X = data['X']
    y = data['y']
    
    issues = []
    
    # 1. 데이터 범위 확인
    if X.min() < 0 or X.max() > 1:
        issues.append(f"❌ 정규화 문제: 데이터 범위가 0-1을 벗어남 (실제: {X.min():.3f} ~ {X.max():.3f})")
    else:
        print("✅ 정규화: 데이터가 0-1 범위 내에 있음")
    
    # 2. NaN/Inf 값 확인
    if np.isnan(X).any():
        issues.append("❌ NaN 값이 존재함")
    else:
        print("✅ NaN 값: 없음")
    
    if np.isinf(X).any():
        issues.append("❌ 무한대 값이 존재함")
    else:
        print("✅ 무한대 값: 없음")
    
    # 3. 클래스 불균형 확인
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) != 3:
        issues.append(f"❌ 클래스 수가 3개가 아님 (실제: {len(unique)}개)")
    else:
        print("✅ 클래스 수: 3개")
        
        # 클래스별 샘플 수 확인
        for label_num, count in zip(unique, counts):
            label_name = list(LABEL_MAP.keys())[list(LABEL_MAP.values()).index(label_num)]
            print(f"   {label_name}: {count}개")
    
    # 4. 특징 수 확인
    expected_features = 25 * 7 + 3  # 25 joints * 7 values + 3 gaze = 178
    if X.shape[2] != expected_features:
        issues.append(f"❌ 특징 수가 예상과 다름 (예상: {expected_features}, 실제: {X.shape[2]})")
        print(f"   💡 실제 데이터는 185개 특징을 가지고 있습니다.")
        print(f"   💡 이는 26개 관절일 가능성이 높습니다 (26 * 7 + 3 = 185)")
    else:
        print(f"✅ 특징 수: {X.shape[2]}개 (예상과 일치)")
    
    # 5. 시퀀스 길이 확인
    if X.shape[1] != MAX_SEQUENCE_LENGTH:
        issues.append(f"❌ 시퀀스 길이가 예상과 다름 (예상: {MAX_SEQUENCE_LENGTH}, 실제: {X.shape[1]})")
    else:
        print(f"✅ 시퀀스 길이: {X.shape[1]}개 (예상과 일치)")
    
    # 문제점 출력
    if issues:
        print("\n🚨 발견된 문제점들:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("\n🎉 데이터 품질 검증 통과!")

# --- 메인 실행 ---
if __name__ == "__main__":
    print("🔍 데이터 전처리 디버깅 시작")
    print("=" * 50)
    
    # 1. 원본 데이터 분석
    analyze_raw_data()
    
    # 2. 전처리된 데이터 분석
    analyze_preprocessed_data()
    
    # 3. 데이터 품질 검증
    validate_data_quality()
    
    print("\n" + "=" * 50)
    print("🔍 디버깅 완료!")
