import json
import os
import glob
from collections import defaultdict

def validate_json_file(file_path):
    """JSON 파일의 유효성을 검사합니다."""
    filename = os.path.basename(file_path)
    issues = []
    
    try:
        # 파일 크기 확인
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            issues.append("파일이 비어있음")
            return filename, False, issues
        
        # JSON 파싱 테스트
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 기본 구조 확인
        if not isinstance(data, dict):
            issues.append("데이터가 딕셔너리가 아님")
            return filename, False, issues
        
        # 필수 키 확인
        if 'frames' not in data:
            issues.append("'frames' 키가 없음")
            return filename, False, issues
        
        frames = data['frames']
        if not isinstance(frames, list):
            issues.append("'frames'가 리스트가 아님")
            return filename, False, issues
        
        if len(frames) == 0:
            issues.append("'frames'가 비어있음")
            return filename, False, issues
        
        # 첫 번째 프레임 구조 확인
        first_frame = frames[0]
        if not isinstance(first_frame, dict):
            issues.append("첫 번째 프레임이 딕셔너리가 아님")
            return filename, False, issues
        
        # 프레임 데이터 구조 확인
        expected_keys = ['timestamp', 'joint_positions', 'gaze_data']
        missing_keys = [key for key in expected_keys if key not in first_frame]
        if missing_keys:
            issues.append(f"필수 키 누락: {missing_keys}")
        
        # 데이터 품질 확인
        if len(frames) < 10:
            issues.append(f"프레임 수가 너무 적음: {len(frames)}")
        
        return filename, True, issues
        
    except json.JSONDecodeError as e:
        issues.append(f"JSON 파싱 오류: {str(e)}")
        return filename, False, issues
    except UnicodeDecodeError as e:
        issues.append(f"인코딩 오류: {str(e)}")
        return filename, False, issues
    except Exception as e:
        issues.append(f"예상치 못한 오류: {str(e)}")
        return filename, False, issues

def main():
    print("🔍 JSON 파일 유효성 검사 시작")
    print("=" * 50)
    
    # 모든 JSON 파일 찾기
    json_files = glob.glob("Data/*.json")
    print(f"📁 총 {len(json_files)}개의 JSON 파일을 찾았습니다.")
    
    if len(json_files) == 0:
        print("❌ JSON 파일을 찾을 수 없습니다!")
        return
    
    # 파일별 유효성 검사
    valid_files = []
    invalid_files = []
    file_stats = defaultdict(int)
    
    print(f"\n🔍 파일별 유효성 검사 중...")
    for i, file_path in enumerate(json_files, 1):
        if i % 10 == 0:
            print(f"   진행률: {i}/{len(json_files)} ({i/len(json_files)*100:.1f}%)")
        
        filename, is_valid, issues = validate_json_file(file_path)
        
        if is_valid:
            valid_files.append(file_path)
            file_stats['valid'] += 1
        else:
            invalid_files.append((file_path, issues))
            file_stats['invalid'] += 1
    
    # 결과 요약
    print(f"\n📊 검사 결과 요약:")
    print(f"   ✅ 유효한 파일: {file_stats['valid']}개")
    print(f"   ❌ 유효하지 않은 파일: {file_stats['invalid']}개")
    print(f"   📈 성공률: {file_stats['valid']/len(json_files)*100:.1f}%")
    
    # 유효하지 않은 파일들 상세 정보
    if invalid_files:
        print(f"\n⚠️ 유효하지 않은 파일들:")
        for file_path, issues in invalid_files[:20]:  # 처음 20개만 출력
            filename = os.path.basename(file_path)
            print(f"   📄 {filename}:")
            for issue in issues:
                print(f"      - {issue}")
        
        if len(invalid_files) > 20:
            print(f"      ... 및 {len(invalid_files) - 20}개 더")
    
    # 라벨별 유효한 파일 개수
    print(f"\n🏷️ 라벨별 유효한 파일 개수:")
    label_counts = defaultdict(int)
    for file_path in valid_files:
        filename = os.path.basename(file_path)
        if filename.startswith("Gaze_"):
            filename = filename[5:]
        label_str = filename.split('_')[0]
        label_counts[label_str] += 1
    
    for label, count in sorted(label_counts.items()):
        print(f"   {label}: {count}개")
    
    # 권장사항
    print(f"\n💡 권장사항:")
    if file_stats['invalid'] > 0:
        print(f"   - {file_stats['invalid']}개 파일의 문제를 해결하세요")
        print(f"   - 손상된 파일은 복구하거나 제거하세요")
    
    if len(label_counts) < 2:
        print(f"   - 최소 2개 이상의 클래스가 필요합니다")
        print(f"   - 현재 {len(label_counts)}개 클래스만 발견됨")
    
    print(f"\n✅ 검사 완료!")

if __name__ == "__main__":
    main()
