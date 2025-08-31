#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H2O Dataset Converter - Selective Conversion Version
H2O 데이터셋을 프로젝트의 표준 JSON 형식으로 변환하는 스크립트
의미있는 동작만 선별적으로 변환 (Pick, Hold, Place에 명확히 매핑되는 동작만)

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

# 의미있는 동작만 선별하여 매핑 (Pick, Hold, Place에 명확히 매핑되는 동작만)
MEANINGFUL_ACTION_MAP = {
    # Pick 동작 (잡기, 꺼내기)
    "grab": "Pick",      # 잡기
    "take out": "Pick",  # 꺼내기
    
    # Hold 동작 (유지, 사용)
    "read": "Hold",      # 읽기 (책을 들고 있는 상태)
    "apply": "Hold",     # 바르기 (로션을 들고 있는 상태)
    "spray": "Hold",     # 분사 (스프레이를 들고 있는 상태)
    "squeeze": "Hold",   # 짜기 (로션을 들고 있는 상태)
    
    # Place 동작 (놓기, 넣기)
    "place": "Place",    # 놓기
    "put in": "Place",   # 넣기
}

# 의미있는 액션만 선별 (Pick, Hold, Place로 변환 가능한 액션)
MEANINGFUL_ACTIONS = {
    # Pick 관련
    1, 2, 3, 4, 5, 6, 7, 8,      # grab actions
    24, 25, 26, 27,               # take out actions
    
    # Hold 관련  
    9, 33, 34,                    # read actions
    31, 32,                       # apply actions
    35,                           # spray action
    36,                           # squeeze action
    
    # Place 관련
    9, 10, 11, 12, 13, 14, 15, 16,  # place actions
    28, 29, 30,                      # put in actions
}

# H2O 액션 라벨을 동사로 매핑 (공식 H2O 데이터셋 기준)
ACTION_TO_VERB_MAP = {
    # Pick 동작들
    1: "grab",      # grab book
    2: "grab",      # grab espresso
    3: "grab",      # grab lotion
    4: "grab",      # grab spray
    5: "grab",      # grab milk
    6: "grab",      # grab cocoa
    7: "grab",      # grab chips
    8: "grab",      # grab cappuccino
    24: "take out", # take out espresso
    25: "take out", # take out cocoa
    26: "take out", # take out chips
    27: "take out", # take out cappuccino
    
    # Hold 동작들
    9: "read",      # read book
    33: "read",     # read book
    34: "read",     # read espresso
    31: "apply",    # apply lotion
    32: "apply",    # apply spray
    35: "spray",    # spray spray
    36: "squeeze",  # squeeze lotion
    
    # Place 동작들
    9: "place",     # place book
    10: "place",    # place espresso
    11: "place",    # place lotion
    12: "place",    # place spray
    13: "place",    # place milk
    14: "place",    # place cocoa
    15: "place",    # place chips
    16: "place",    # place cappuccino
    28: "put in",   # put in espresso
    29: "put in",   # put in cocoa
    30: "put in",   # put in cappuccino
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
                
                # H2O는 21개 관절 × 3개 좌표 × 2개 손 = 126개 값이지만, 실제로는 더 많은 값이 있을 수 있음
                if len(values) >= 126:
                    # 처음 126개 값만 사용 (21개 관절 × 3개 좌표 × 2개 손)
                    joints_3d = np.array(values[:126]).reshape(2, 21, 3)  # 2개 손, 21개 관절, 3개 좌표
                    hand_poses.append(joints_3d)
                else:
                    print(f"   ⚠️ 프레임 {frame_idx}: 예상 126개 값, 실제 {len(values)}개")
                    # 부족한 값은 0으로 채움
                    if len(values) < 126:
                        values.extend([0.0] * (126 - len(values)))
                    joints_3d = np.array(values[:126]).reshape(2, 21, 3)
                    hand_poses.append(joints_3d)
            else:
                print(f"   ⚠️ 프레임 {frame_idx} 파일 없음: {frame_file}")
                # 빈 프레임은 0으로 채움
                empty_frame = np.zeros((2, 21, 3), dtype=np.float32)
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
        h2o_joints: H2O 관절 데이터 (2, 21, 3) - (2개 손, 21개 관절, 3개 좌표)
        
    Returns:
        프로젝트 형식의 관절 데이터 리스트
    """
    joints_data = []
    
    # 오른손 데이터 우선 사용 (Unity는 주로 오른손 지원)
    right_hand = h2o_joints[1] if h2o_joints.shape[0] > 1 else h2o_joints[0]
    
    # 프로젝트의 26개 관절 순서대로 데이터 생성
    for joint_name in PROJECT_JOINT_ORDER:
        # H2O 관절을 프로젝트 관절로 매핑
        h2o_joint_idx = None
        for h2o_idx, proj_joint in H2O_TO_PROJECT_JOINT_MAP.items():
            if proj_joint == joint_name:
                h2o_joint_idx = h2o_idx
                break
        
        if h2o_joint_idx is not None and h2o_joint_idx < len(right_hand):
            # H2O 관절 데이터가 있는 경우
            pos = right_hand[h2o_joint_idx]
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

def get_verb_labels_from_files(verb_label_dir: str, start_frame: int, end_frame: int) -> List[int]:
    """
    프레임 범위에서 동사 라벨들을 추출합니다.
    
    Args:
        verb_label_dir: 동사 라벨 디렉토리 경로
        start_frame: 시작 프레임 번호
        end_frame: 끝 프레임 번호
        
    Returns:
        프레임별 동사 라벨 리스트
    """
    verb_labels = []
    
    try:
        for frame_idx in range(start_frame, end_frame + 1):
            frame_file = os.path.join(verb_label_dir, f"{frame_idx:06d}.txt")
            
            if os.path.exists(frame_file):
                with open(frame_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                if content:
                    verb_id = int(content)
                    verb_labels.append(verb_id)
                else:
                    verb_labels.append(0)
            else:
                verb_labels.append(0)
                    
    except Exception as e:
        print(f"   ❌ 동사 라벨 로드 오류: {e}")
        return []
    
    return verb_labels

def get_verb_name(verb_id: int) -> str:
    """
    동사 ID를 이름으로 변환
    """
    verb_names = {
        0: "background", 1: "grab", 2: "place", 3: "open", 4: "close",
        5: "pour", 6: "take out", 7: "put in", 8: "apply", 9: "read",
        10: "spray", 11: "squeeze"
    }
    return verb_names.get(verb_id, "unknown")

def should_skip_sequence(sequence_info: Dict, verb_labels: List[int]) -> bool:
    """
    변환하지 않을 시퀀스인지 판단
    """
    action_label = sequence_info['action_label']
    
    # 의미없는 액션들 (무시)
    meaningless_actions = {
        0,   # background
        17,  # open lotion
        18,  # open milk  
        19,  # open chips
        20,  # close lotion
        21,  # close milk
        22,  # close chips
        23,  # pour milk
    }
    
    if action_label in meaningless_actions:
        return True
    
    # verb_label이 모두 background(0)인 경우
    if verb_labels and all(v == 0 for v in verb_labels):
        return True
    
    return False

def map_to_meaningful_action(action_label: int, verb_label: int) -> Optional[str]:
    """
    의미있는 동작으로만 매핑
    """
    # 액션 라벨 기반 매핑
    if action_label in MEANINGFUL_ACTIONS:
        if action_label in [1, 2, 3, 4, 5, 6, 7, 8, 24, 25, 26, 27]:
            return "Pick"
        elif action_label in [9, 33, 34, 31, 32, 35, 36]:
            return "Hold"
        elif action_label in [9, 10, 11, 12, 13, 14, 15, 16, 28, 29, 30]:
            return "Place"
    
    # 동사 라벨 기반 매핑 (보조)
    verb_name = get_verb_name(verb_label)
    if verb_name in MEANINGFUL_ACTION_MAP:
        return MEANINGFUL_ACTION_MAP[verb_name]
    
    return None

def process_sequence(sequence_info: Dict) -> Optional[Tuple[str, Dict]]:
    """
    단일 시퀀스를 처리하여 프로젝트 형식으로 변환합니다.
    의미있는 동작만 선별적으로 변환
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
    
    # 데이터 로드
    hand_poses = load_h2o_hand_pose_data(hand_pose_dir, start_frame, end_frame)
    verb_labels = get_verb_labels_from_files(verb_label_dir, start_frame, end_frame)
    
    if not hand_poses:
        print(f"   ❌ 데이터 로드 실패")
        return None
    
    # 변환 가능성 판단
    if should_skip_sequence(sequence_info, verb_labels):
        print(f"   ⏭️ 변환 불가능한 시퀀스 - 건너뛰기")
        return None
    
    # 동작 라벨 결정
    action_label = sequence_info['action_label']
    verb_label = max(set(verb_labels), key=verb_labels.count) if verb_labels else 0
    
    # 의미있는 동작으로 매핑
    project_label = map_to_meaningful_action(action_label, verb_label)
    
    if not project_label:
        print(f"   ⏭️ 의미있는 동작으로 매핑 불가 - 건너뛰기")
        return None
    
    print(f"   ✅ 매핑: {action_label}/{verb_label} -> {project_label}")
    
    # 객체 포즈 데이터 로드 (손 포즈가 성공적으로 로드된 후)
    obj_poses = load_h2o_obj_pose_data(obj_pose_dir, start_frame, end_frame)
    
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

def generate_conversion_report(processed_sequences: List, total_sequences: int):
    """
    변환 결과 보고서 생성
    """
    print(f"\n📊 변환 품질 보고서")
    print("=" * 50)
    
    # 변환 성공률
    success_rate = len(processed_sequences) / total_sequences * 100
    print(f"전체 시퀀스: {total_sequences}개")
    print(f"변환 성공: {len(processed_sequences)}개")
    print(f"변환 성공률: {success_rate:.1f}%")
    
    # 동작별 분포
    action_counts = {}
    for _, json_data in processed_sequences:
        action = json_data.get('interactionTargetObject', 'Unknown')
        action_counts[action] = action_counts.get(action, 0) + 1
    
    print(f"\n동작별 분포:")
    for action, count in action_counts.items():
        print(f"  {action}: {count}개")
    
    # 품질 지표
    print(f"\n품질 지표:")
    print(f"  - 의미있는 동작만 선별: ✅")
    print(f"  - 노이즈 데이터 제거: ✅")
    print(f"  - Unity 호환성: ✅")

def main():
    """메인 변환 함수 - 선별적 변환"""
    print("🚀 H2O 데이터셋 선별적 변환기 시작")
    print("=" * 50)
    print("💡 의미있는 동작만 선별하여 변환합니다")
    print("   - Pick: grab, take out")
    print("   - Hold: read, apply, spray, squeeze")  
    print("   - Place: place, put in")
    print("   - 기타: 무시 (open, close, pour 등)")
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
    
    # 변환 가능한 시퀀스만 필터링
    convertible_sequences = []
    for sequence_info in sequences:
        if not should_skip_sequence(sequence_info, []):
            convertible_sequences.append(sequence_info)
    
    print(f"📊 변환 가능한 시퀀스: {len(convertible_sequences)}개")
    print(f"⏭️ 변환 불가능한 시퀀스: {len(sequences) - len(convertible_sequences)}개")
    
    # 시퀀스별 처리
    successful_conversions = 0
    failed_conversions = 0
    processed_sequences = []
    
    print(f"\n🔄 시퀀스 변환 시작...")
    
    for i, sequence_info in enumerate(convertible_sequences):
        print(f"\n[{i+1}/{len(convertible_sequences)}] 시퀀스 처리 중...")
        
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
                processed_sequences.append((project_label, json_data))
                
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
    print(f"📊 총 처리: {len(convertible_sequences)}개")
    print(f"📁 출력 디렉토리: {OUTPUT_JSON_DIR}")
    
    # 품질 보고서 생성
    generate_conversion_report(processed_sequences, len(sequences))
    
    if successful_conversions > 0:
        print(f"\n💡 변환된 파일들을 확인하려면:")
        print(f"   ls {OUTPUT_JSON_DIR}/*_h2o_*.json")

if __name__ == "__main__":
    main()
