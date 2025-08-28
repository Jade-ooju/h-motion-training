import glob
import os

def extract_label_from_filename(filename):
    """파일명에서 동작 라벨을 추출합니다."""
    # Gaze_ 접두사가 있는 경우 제거
    if filename.startswith("Gaze_"):
        filename = filename[5:]  # "Gaze_" 제거
    
    # 첫 번째 언더스코어 이전의 부분을 라벨로 사용
    label_str = filename.split('_')[0]
    return label_str

# 라벨 매핑
LABEL_MAP = {"Pick": 0, "Hold": 1, "Place": 2}

# 파일 목록 가져오기
json_files = glob.glob("Data/*.json")
print(f"총 {len(json_files)}개의 JSON 파일을 찾았습니다.")

# 라벨별 파일 개수 확인
label_counts = {}
for file_path in json_files:
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

# 처음 20개 파일의 라벨 확인
print(f"\n🔍 처음 20개 파일의 라벨:")
for file_path in json_files[:20]:
    filename = os.path.basename(file_path)
    label_str = extract_label_from_filename(filename)
    if label_str in LABEL_MAP:
        print(f"   {filename} -> {label_str} -> {LABEL_MAP[label_str]}")
    else:
        print(f"   {filename} -> {label_str} -> ❌ 알 수 없는 라벨")
