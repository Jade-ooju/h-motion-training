#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H2O Dataset Converter
H2O 데이터셋을 프로젝트의 표준 JSON 형식으로 변환하는 스크립트

H2O 데이터셋 구조:
- label_split/action_train.txt: 시퀀스별 액션 라벨 정보
- subject*/h*/*/cam4/hand_pose/: 손 포즈 데이터 (프레임별 .txt)
- subject*/h*/*/cam4/obj_pose/: 객체 포즈 데이터 (프레임별 .txt)
- subject*/h*/*/cam4/verb_label/: 동사 라벨 (프레임별 .txt)

출력 형식:
- Data/ 폴더에 [매핑된라벨]_h2o_[시퀀스ID].json 형식으로 저장
"""

import os
import json
import numpy as np
import glob
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import re

# --- 1. 설정 변수 정의 ---
H2O_DATA_ROOT = "../../H2O_Dataset"  # H2O 데이터셋 루트 경로
OUTPUT_JSON_DIR = "../../Data"        # 출력 JSON 파일 저장 경로

# 동사 라벨을 프로젝트 라벨로 매핑
VERB_LABEL_MAP = {
    # Pick 동작
    "grab": "Pick",
    "take": "Pick",
    "pick": "Pick",
    "grasp": "Pick",
    "lift": "Pick",
    "raise": "Pick",
    
    # Hold 동작  
    "read": "Hold",
    "squeeze": "Hold",
    "hold": "Hold",
    "carry": "Hold",
    "support": "Hold",
    "maintain": "Hold",
    
    # Place 동작
    "place": "Place", 
    "put": "Place",
    "drop": "Place",
    "release": "Place",
    "set": "Place",
    "position": "Place",
    "move": "Place"
}

# H2O 액션 라벨을 동사로 매핑 (H2O 데이터셋 표준)
ACTION_TO_VERB_MAP = {
    # Pick 동작들
    1: "grab",      # grab
    2: "take",      # take
    3: "pick",      # pick
    4: "grasp",     # grasp
    5: "lift",      # lift
    6: "raise",     # raise
    
    # Hold 동작들
    7: "hold",      # hold
    8: "carry",     # carry
    9: "support",   # support
    10: "maintain", # maintain
    11: "read",     # read
    12: "squeeze",  # squeeze
    
    # Place 동작들
    13: "place",    # place
    14: "put",      # put
    15: "drop",     # drop
    16: "release",  # release
    17: "set",      # set
    18: "position", # position
    19: "move",     # move
    
    # 기타 동작들 (매핑되지 않는 것들)
    20: "hold",     # 기타 -> Hold로 매핑
    21: "hold",     # 기타 -> Hold로 매핑
    22: "hold",     # 기타 -> Hold로 매핑
    23: "hold",     # 기타 -> Hold로 매핑
    24: "hold",     # 기타 -> Hold로 매핑
    25: "hold",     # 기타 -> Hold로 매핑
    26: "hold",     # 기타 -> Hold로 매핑
    27: "hold",     # 기타 -> Hold로 매핑
    28: "hold",     # 기타 -> Hold로 매핑
    29: "hold",     # 기타 -> Hold로 매핑
    30: "hold",     # 기타 -> Hold로 매핑
    31: "hold",     # 기타 -> Hold로 매핑
    32: "hold",     # 기타 -> Hold로 매핑
    33: "hold",     # 기타 -> Hold로 매핑
    34: "hold",     # 기타 -> Hold로 매핑
    35: "hold",     # 기타 -> Hold로 매핑
    36: "hold",     # 기타 -> Hold로 매핑
}

# 프로젝트의 관절 순서 (data_preprocessor.py와 동일)
PROJECT_JOINT_ORDER = [
    "Wrist", "ForearmWrist", "Palm", "ThumbMetacarpal", "ThumbProximal", "ThumbDistal", "ThumbTip",
    "IndexMetacarpal", "IndexProximal", "IndexIntermediate", "IndexDistal", "IndexTip",
    "MiddleMetacarpal", "MiddleProximal", "MiddleIntermediate", "MiddleDistal", "MiddleTip",
    "RingMetacarpal", "RingProximal", "RingIntermediate", "RingDistal", "RingTip",
    "PinkyMetacarpal", "PinkyProximal", "PinkyIntermediate", "PinkyDistal"
]

# H2O의 21개 관절을 프로젝트 관절로 매핑 (가장 유사한 관절로 매핑)
H2O_TO_PROJECT_JOINT_MAP = {
    # H2O의 21개 관절을 프로젝트의 26개 관절로 매핑
    # 실제 매핑은 H2O 데이터의 관절 구조를 보고 조정 필요
    0: "Wrist",           # H2O joint 0 -> Wrist
    1: "ThumbMetacarpal", # H2O joint 1 -> ThumbMetacarpal
    2: "ThumbProximal",   # H2O joint 2 -> ThumbProximal
    3: "ThumbDistal",     # H2O joint 3 -> ThumbDistal
    4: "ThumbTip",        # H2O joint 4 -> ThumbTip
    5: "IndexMetacarpal", # H2O joint 5 -> IndexMetacarpal
    6: "IndexProximal",   # H2O joint 6 -> IndexProximal
    7: "IndexIntermediate", # H2O joint 7 -> IndexIntermediate
    8: "IndexDistal",     # H2O joint 8 -> IndexDistal
    9: "IndexTip",        # H2O joint 9 -> IndexTip
    10: "MiddleMetacarpal", # H2O joint 10 -> MiddleMetacarpal
    11: "MiddleProximal", # H2O joint 11 -> MiddleProximal
    12: "MiddleIntermediate", # H2O joint 12 -> MiddleIntermediate
    13: "MiddleDistal",   # H2O joint 13 -> MiddleDistal
    14: "MiddleTip",      # H2O joint 14 -> MiddleTip
    15: "RingMetacarpal", # H2O joint 15 -> RingMetacarpal
    16: "RingProximal",   # H2O joint 16 -> RingProximal
    17: "RingIntermediate", # H2O joint 17 -> RingIntermediate
    18: "RingDistal",     # H2O joint 18 -> RingDistal
    19: "RingTip",        # H2O joint 19 -> RingTip
    20: "PinkyMetacarpal", # H2O joint 20 -> PinkyMetacarpal
}

# --- 2. 유틸리티 함수들 ---

def extract_label_from_filename(filename: str) -> str:
    """파일명에서 동작 라벨을 추출합니다."""
    if filename.startswith("Gaze_"):
        filename = filename[5:]
    label_str = filename.split('_')[0]
    return label_str

def parse_action_labels_file(file_path: str) -> List[Dict]:
    """
    action_train.txt 파일을 파싱하여 시퀀스 정보를 추출합니다.
    
    파일 형식: id path action_label start_act end_act start_frame end_frame
    """
    sequences = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # 헤더 건너뛰기
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) >= 7:
                sequence_info = {
                    'id': int(parts[0]),
                    'path': parts[1],
                    'action_label': int(parts[2]),
                    'start_act': int(parts[3]),
                    'end_act': int(parts[4]),
                    'start_frame': int(parts[5]),
                    'end_frame': int(parts[6])
                }
                sequences.append(sequence_info)
                
    except Exception as e:
        print(f"❌ 액션 라벨 파일 파싱 오류: {e}")
        return []
    
    return sequences

def load_h2o_hand_pose_data(hand_pose_dir: str, start_frame: int, end_frame: int) -> List[np.ndarray]:
    """
    H2O 손 포즈 데이터를 로드합니다.
    
    Args:
        hand_pose_dir: 손 포즈 데이터 디렉토리 경로
        start_frame: 시작 프레임 번호
        end_frame: 끝 프레임 번호
        
    Returns:
        프레임별 손 포즈 데이터 리스트
    """
    hand_poses = []
    
    try:
        for frame_idx in range(start_frame, end_frame + 1):
            frame_file = os.path.join(hand_pose_dir, f"{frame_idx:06d}.txt")
            
            if os.path.exists(frame_file):
                with open(frame_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                # 공백으로 구분된 숫자들을 파싱
                values = [float(x) for x in content.split()]
                
                # H2O는 21개 관절 × 3개 좌표 = 63개 값이지만, 실제로는 더 많은 값이 있을 수 있음
                if len(values) >= 63:
                    # 처음 63개 값만 사용 (21개 관절 × 3개 좌표)
                    joints_3d = np.array(values[:63]).reshape(21, 3)
                    hand_poses.append(joints_3d)
                else:
                    print(f"   ⚠️ 프레임 {frame_idx}: 예상 63개 값, 실제 {len(values)}개")
                    # 부족한 값은 0으로 채움
                    if len(values) < 63:
                        values.extend([0.0] * (63 - len(values)))
                    joints_3d = np.array(values[:63]).reshape(21, 3)
                    hand_poses.append(joints_3d)
            else:
                print(f"   ⚠️ 프레임 {frame_idx} 파일 없음: {frame_file}")
                # 빈 프레임은 0으로 채움
                empty_frame = np.zeros((21, 3), dtype=np.float32)
                hand_poses.append(empty_frame)
                
    except Exception as e:
        print(f"   ❌ 손 포즈 데이터 로드 오류: {e}")
        return []
    
    return hand_poses

def load_h2o_obj_pose_data(obj_pose_dir: str, start_frame: int, end_frame: int) -> List[Dict]:
    """
    H2O 객체 포즈 데이터를 로드합니다.
    
    Args:
        obj_pose_dir: 객체 포즈 데이터 디렉토리 경로
        start_frame: 시작 프레임 번호
        end_frame: 끝 프레임 번호
        
    Returns:
        프레임별 객체 포즈 데이터 리스트
    """
    obj_poses = []
    
    try:
        for frame_idx in range(start_frame, end_frame + 1):
            frame_file = os.path.join(obj_pose_dir, f"{frame_idx:06d}.txt")
            
            if os.path.exists(frame_file):
                with open(frame_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                # 공백으로 구분된 숫자들을 파싱
                values = [float(x) for x in content.split()]
                
                # 첫 번째 값은 객체 ID, 나머지는 3D 좌표들
                if len(values) > 1:
                    obj_id = int(values[0])
                    coords = values[1:]  # 3D 좌표들
                    
                    # 3개씩 묶어서 (x, y, z) 좌표로 재구성
                    if len(coords) % 3 == 0:
                        points_3d = np.array(coords).reshape(-1, 3)
                        obj_poses.append({
                            'obj_id': obj_id,
                            'points': points_3d
                        })
                    else:
                        print(f"   ⚠️ 프레임 {frame_idx}: 3의 배수가 아닌 좌표 개수 {len(coords)}")
                        obj_poses.append({
                            'obj_id': obj_id,
                            'points': np.zeros((1, 3), dtype=np.float32)
                        })
                else:
                    print(f"   ⚠️ 프레임 {frame_idx}: 데이터 부족")
                    obj_poses.append({
                        'obj_id': 0,
                        'points': np.zeros((1, 3), dtype=np.float32)
                    })
            else:
                print(f"   ⚠️ 프레임 {frame_idx} 파일 없음: {frame_file}")
                obj_poses.append({
                    'obj_id': 0,
                    'points': np.zeros((1, 3), dtype=np.float32)
                })
                
    except Exception as e:
        print(f"   ❌ 객체 포즈 데이터 로드 오류: {e}")
        return []
    
    return obj_poses

def map_h2o_joints_to_project_format(h2o_joints: np.ndarray) -> List[Dict]:
    """
    H2O의 21개 관절 데이터를 프로젝트의 26개 관절 형식으로 변환합니다.
    
    Args:
        h2o_joints: H2O 관절 데이터 (21, 3) - (x, y, z) 좌표
        
    Returns:
        프로젝트 형식의 관절 데이터 리스트
    """
    joints_data = []
    
    # 프로젝트의 26개 관절 순서대로 데이터 생성
    for joint_name in PROJECT_JOINT_ORDER:
        # H2O 관절을 프로젝트 관절로 매핑
        h2o_joint_idx = None
        for h2o_idx, proj_joint in H2O_TO_PROJECT_JOINT_MAP.items():
            if proj_joint == joint_name:
                h2o_joint_idx = h2o_idx
                break
        
        if h2o_joint_idx is not None and h2o_joint_idx < len(h2o_joints):
            # H2O 관절 데이터가 있는 경우
            pos = h2o_joints[h2o_joint_idx]
            joints_data.append({
                "jointName": joint_name,
                "position": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},  # H2O에는 회전 정보가 없음
                "confidence": 1.0
            })
        else:
            # 매핑되지 않는 관절은 0으로 채움
            joints_data.append({
                "jointName": joint_name,
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                "confidence": 0.0
            })
    
    return joints_data

def calculate_interaction_target_position(obj_poses: List[Dict]) -> Dict[str, float]:
    """
    객체 포즈 데이터에서 상호작용 대상 위치를 계산합니다.
    
    Args:
        obj_poses: 프레임별 객체 포즈 데이터 리스트
        
    Returns:
        평균 3D 좌표
    """
    if not obj_poses:
        return {"x": 0.0, "y": 0.0, "z": 0.0}
    
    all_points = []
    for obj_pose in obj_poses:
        if 'points' in obj_pose and len(obj_pose['points']) > 0:
            all_points.extend(obj_pose['points'])
    
    if not all_points:
        return {"x": 0.0, "y": 0.0, "z": 0.0}
    
    # 모든 포인트의 평균 계산
    all_points = np.array(all_points)
    mean_position = np.mean(all_points, axis=0)
    
    return {
        "x": float(mean_position[0]),
        "y": float(mean_position[1]),
        "z": float(mean_position[2])
    }

def convert_h2o_sequence_to_project_format(
    sequence_info: Dict,
    hand_poses: List[np.ndarray],
    obj_poses: List[Dict]
) -> Dict:
    """
    H2O 시퀀스 데이터를 프로젝트의 JSON 형식으로 변환합니다.
    
    Args:
        sequence_info: 시퀀스 정보
        hand_poses: 프레임별 손 포즈 데이터
        obj_poses: 프레임별 객체 포즈 데이터
        
    Returns:
        프로젝트 형식의 JSON 데이터
    """
    # 상호작용 대상 위치 계산
    interaction_target_position = calculate_interaction_target_position(obj_poses)
    
    # 프레임 데이터 생성
    frames = []
    for frame_idx, (hand_pose, obj_pose) in enumerate(zip(hand_poses, obj_poses)):
        # H2O 관절을 프로젝트 형식으로 변환
        joints = map_h2o_joints_to_project_format(hand_pose)
        
        frame_data = {
            "timestamp": float(frame_idx),
            "isRightHand": True,  # H2O는 주로 오른손 데이터
            "joints": joints
        }
        frames.append(frame_data)
    
    # JSON 구조 생성
    json_data = {
        "sessionId": f"h2o_{sequence_info['path'].replace('/', '_')}",
        "startTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "interactionTargetObject": f"Object_{sequence_info['action_label']}",
        "interactionTargetPosition": interaction_target_position,
        "frames": frames
    }
    
    return json_data

def get_verb_label_from_files(verb_label_dir: str, start_frame: int, end_frame: int) -> Optional[str]:
    """
    프레임 범위에서 가장 많이 나타나는 동사 라벨을 찾습니다.
    
    Args:
        verb_label_dir: 동사 라벨 디렉토리 경로
        start_frame: 시작 프레임 번호
        end_frame: 끝 프레임 번호
        
    Returns:
        가장 많이 나타나는 동사 라벨
    """
    verb_counts = {}
    
    try:
        for frame_idx in range(start_frame, end_frame + 1):
            frame_file = os.path.join(verb_label_dir, f"{frame_idx:06d}.txt")
            
            if os.path.exists(frame_file):
                with open(frame_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                if content:
                    verb_id = int(content)
                    verb_counts[verb_id] = verb_counts.get(verb_id, 0) + 1
                    
    except Exception as e:
        print(f"   ❌ 동사 라벨 로드 오류: {e}")
        return None
    
    if not verb_counts:
        return None
    
    # 가장 많이 나타나는 동사 ID 찾기
    most_common_verb_id = max(verb_counts, key=verb_counts.get)
    
    # 동사 ID를 실제 동사로 변환 (H2O 데이터셋의 동사 매핑 필요)
    # 여기서는 간단하게 action_label을 사용
    return most_common_verb_id

def map_action_label_to_verb(action_label: int) -> Optional[str]:
    """
    H2O 액션 라벨을 동사로 매핑합니다.
    H2O 데이터셋의 표준 액션 라벨 매핑 사용
    """
    return ACTION_TO_VERB_MAP.get(action_label)

def process_sequence(sequence_info: Dict) -> Optional[Tuple[str, Dict]]:
    """
    단일 시퀀스를 처리하여 프로젝트 형식으로 변환합니다.
    
    Args:
        sequence_info: 시퀀스 정보
        
    Returns:
        (매핑된 라벨, JSON 데이터) 튜플 또는 None
    """
    sequence_path = sequence_info['path']
    start_frame = sequence_info['start_frame']
    end_frame = sequence_info['end_frame']
    
    print(f"   🔄 처리 중: {sequence_path} (프레임 {start_frame}-{end_frame})")
    
    # 데이터 경로 구성
    hand_pose_dir = os.path.join(H2O_DATA_ROOT, sequence_path, "cam4", "hand_pose")
    obj_pose_dir = os.path.join(H2O_DATA_ROOT, sequence_path, "cam4", "obj_pose")
    verb_label_dir = os.path.join(H2O_DATA_ROOT, sequence_path, "cam4", "verb_label")
    
    # 경로 존재 확인
    if not os.path.exists(hand_pose_dir):
        print(f"   ❌ 손 포즈 디렉토리 없음: {hand_pose_dir}")
        return None
    
    if not os.path.exists(obj_pose_dir):
        print(f"   ❌ 객체 포즈 디렉토리 없음: {obj_pose_dir}")
        return None
    
    # 데이터 로드
    hand_poses = load_h2o_hand_pose_data(hand_pose_dir, start_frame, end_frame)
    obj_poses = load_h2o_obj_pose_data(obj_pose_dir, start_frame, end_frame)
    
    if not hand_poses or not obj_poses:
        print(f"   ❌ 데이터 로드 실패")
        return None
    
    # 동사 라벨 결정 (액션 라벨 우선, 없으면 프레임별 동사 라벨 사용)
    verb_label = map_action_label_to_verb(sequence_info['action_label'])
    
    if not verb_label:
        # 프레임별 동사 라벨에서 가장 많이 나타나는 것 사용
        verb_label = get_verb_label_from_files(verb_label_dir, start_frame, end_frame)
    
    if not verb_label:
        print(f"   ⚠️ 동사 라벨을 찾을 수 없음")
        return None
    
    # 프로젝트 라벨로 매핑
    if verb_label not in VERB_LABEL_MAP:
        print(f"   ⚠️ 매핑되지 않는 동사: {verb_label}")
        return None
    
    project_label = VERB_LABEL_MAP[verb_label]
    print(f"   ✅ 매핑: {verb_label} -> {project_label}")
    
    # JSON 형식으로 변환
    json_data = convert_h2o_sequence_to_project_format(sequence_info, hand_poses, obj_poses)
    
    return (project_label, json_data)

def save_json_file(json_data: Dict, output_path: str):
    """JSON 데이터를 파일로 저장합니다."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
            
        print(f"   💾 저장 완료: {output_path}")
        
    except Exception as e:
        print(f"   ❌ 파일 저장 오류: {e}")

def main():
    """메인 변환 함수"""
    print("🚀 H2O 데이터셋 변환기 시작")
    print("=" * 50)
    
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
    
    # 액션 라벨 파일 경로
    action_labels_file = os.path.join(H2O_DATA_ROOT, "label_split", "action_train.txt")
    
    if not os.path.exists(action_labels_file):
        print(f"❌ 액션 라벨 파일을 찾을 수 없음: {action_labels_file}")
        return
    
    # 액션 라벨 파일 파싱
    print("📖 액션 라벨 파일 파싱 중...")
    sequences = parse_action_labels_file(action_labels_file)
    print(f"📊 총 {len(sequences)}개 시퀀스 발견")
    
    if not sequences:
        print("❌ 파싱된 시퀀스가 없습니다.")
        return
    
    # 시퀀스별 처리
    successful_conversions = 0
    failed_conversions = 0
    
    print(f"\n🔄 시퀀스 변환 시작...")
    
    for i, sequence_info in enumerate(sequences):
        print(f"\n[{i+1}/{len(sequences)}] 시퀀스 처리 중...")
        
        try:
            result = process_sequence(sequence_info)
            
            if result:
                project_label, json_data = result
                
                # 출력 파일명 생성
                sequence_id = sequence_info['path'].replace('/', '_')
                output_filename = f"{project_label}_h2o_{sequence_id}.json"
                output_path = os.path.join(OUTPUT_JSON_DIR, output_filename)
                
                # JSON 파일 저장
                save_json_file(json_data, output_path)
                successful_conversions += 1
                
            else:
                failed_conversions += 1
                
        except Exception as e:
            print(f"   ❌ 시퀀스 처리 오류: {e}")
            failed_conversions += 1
    
    # 결과 요약
    print(f"\n🎉 변환 완료!")
    print("=" * 50)
    print(f"✅ 성공: {successful_conversions}개")
    print(f"❌ 실패: {failed_conversions}개")
    print(f"📊 총 처리: {len(sequences)}개")
    print(f"📁 출력 디렉토리: {OUTPUT_JSON_DIR}")
    
    if successful_conversions > 0:
        print(f"\n💡 변환된 파일들을 확인하려면:")
        print(f"   ls {OUTPUT_JSON_DIR}/*_h2o_*.json")

if __name__ == "__main__":
    main()
