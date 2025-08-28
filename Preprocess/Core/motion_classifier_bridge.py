#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unity Python Bridge for Motion Classification
Unity에서 전송된 관절 데이터를 받아 동작을 분류하는 Python 브리지
Gaze_Hold_010.json 형식 지원 및 향상된 특징 처리
"""

import json
import numpy as np
import joblib
import sys
import os
from typing import Dict, List, Tuple, Optional

class UnityPythonBridge:
    """Unity와 통신하여 동작을 분류하는 Python 브리지"""
    
    def __init__(self, model_path: str = "best_gradient_boosting_model.pkl", 
                 feature_info_path: str = "ensemble_feature_selection_info.json",
                 use_enhanced_features: bool = True):
        """
        Python 브리지 초기화
        
        Args:
            model_path: 훈련된 Random Forest 모델 파일 경로
            feature_info_path: 특징 선택 정보 파일 경로
            use_enhanced_features: 향상된 특징 사용 여부
        """
        try:
            # 모델 로드
            print(f"🔍 모델 로드 중: {model_path}")
            self.model = joblib.load(model_path)
            print(f"✅ 모델 로드 완료: {type(self.model).__name__}")
            
            # 특징 선택 정보 로드
            print(f"🔍 특징 선택 정보 로드 중: {feature_info_path}")
            with open(feature_info_path, 'r', encoding='utf-8') as f:
                self.feature_info = json.load(f)
            print(f"✅ 특징 선택 정보 로드 완료")
            
            # 향상된 특징 사용 여부 설정
            self.use_enhanced_features = use_enhanced_features
            
            # 라벨 매핑
            self.label_map = {0: "Hold", 1: "Pick", 2: "Place"}
            self.reverse_label_map = {v: k for k, v in self.label_map.items()}
            
            print(f"🎯 동작 분류 준비 완료!")
            print(f"   - 모델: {type(self.model).__name__}")
            print(f"   - 총 특징 수: {self.feature_info['total_features']}")
            print(f"   - 선택된 특징: {len(self.feature_info['selected_features'])}")
            print(f"   - 압축률: {self.feature_info['compression_ratio']:.1%}")
            print(f"   - 모델 정확도: {self.feature_info['best_accuracy']:.1%}")
            print(f"   - 향상된 특징 사용: {self.use_enhanced_features}")
            
        except Exception as e:
            print(f"❌ 초기화 오류: {e}")
            raise
    
    def flatten_frame(self, frame: Dict) -> List[float]:
        """
        단일 프레임의 모든 관절/시선 데이터를 1차원 벡터로 변환
        
        Args:
            frame: JSON 프레임 데이터
            
        Returns:
            평탄화된 1차원 벡터
        """
        flat_data = []
        
        # 손 관절 데이터를 순서대로 추가 (순서 유지가 매우 중요!)
        joint_order = [
            "Wrist", "ForearmWrist", "Palm", "ThumbMetacarpal", "ThumbProximal", "ThumbDistal", "ThumbTip",
            "IndexMetacarpal", "IndexProximal", "IndexIntermediate", "IndexDistal", "IndexTip",
            "MiddleMetacarpal", "MiddleProximal", "MiddleIntermediate", "MiddleDistal", "MiddleTip",
            "RingMetacarpal", "RingProximal", "RingIntermediate", "RingDistal", "RingTip",
            "PinkyMetacarpal", "PinkyProximal", "PinkyIntermediate", "PinkyDistal"
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
        
        # Gaze Target Position (x, y, z) - 현재는 0으로 초기화
        # 향후 시선 데이터가 추가되면 여기서 처리
        flat_data.extend([0, 0, 0])
        
        return flat_data
    
    def flatten_frame_with_gaze(self, frame: Dict, session_data: Dict) -> List[float]:
        """
        시선 데이터를 포함한 향상된 프레임 평탄화
        
        Args:
            frame: 프레임 데이터
            session_data: 세션 전체 데이터 (interactionTargetPosition 등 포함)
            
        Returns:
            향상된 특징을 포함한 평탄화된 벡터
        """
        # 기본 관절 데이터 평탄화
        flat_data = self.flatten_frame(frame)
        
        # 시선 데이터 추가 (interactionTargetPosition)
        if 'interactionTargetPosition' in session_data:
            target_pos = session_data['interactionTargetPosition']
            # 기존 gaze 데이터를 실제 target position으로 교체
            flat_data[-3:] = [target_pos.get('x', 0), target_pos.get('y', 0), target_pos.get('z', 0)]
        
        # 추가 특징들
        additional_features = []
        
        # 타임스탬프 정보
        if 'timestamp' in frame:
            timestamp = frame['timestamp']
            additional_features.append(timestamp)
        
        # 오른손/왼손 정보
        if 'isRightHand' in frame:
            additional_features.append(1.0 if frame['isRightHand'] else 0.0)
        
        # 관절 신뢰도 평균
        if 'joints' in frame and frame['joints']:
            confidences = [joint.get('confidence', 0.0) for joint in frame['joints']]
            avg_confidence = np.mean(confidences) if confidences else 0.0
            additional_features.append(avg_confidence)
        
        # 추가 특징을 flat_data에 확장
        flat_data.extend(additional_features)
        
        return flat_data
    
    def preprocess_input_data(self, input_data: Dict) -> np.ndarray:
        """
        Unity에서 전송된 입력 데이터를 전처리
        
        Args:
            input_data: Unity에서 전송된 JSON 데이터
            
        Returns:
            전처리된 특징 벡터
        """
        try:
            if 'frames' not in input_data:
                raise ValueError("입력 데이터에 'frames' 필드가 없습니다.")
            
            # 향상된 특징 사용 여부에 따라 다른 평탄화 함수 사용
            if self.use_enhanced_features:
                sequence = [self.flatten_frame_with_gaze(frame, input_data) for frame in input_data["frames"]]
            else:
                sequence = [self.flatten_frame(frame) for frame in input_data["frames"]]
            
            # 시퀀스 길이 통일 (MAX_SEQUENCE_LENGTH = 300)
            MAX_SEQUENCE_LENGTH = 300
            if len(sequence) > MAX_SEQUENCE_LENGTH:
                sequence = sequence[:MAX_SEQUENCE_LENGTH]
            elif len(sequence) < MAX_SEQUENCE_LENGTH:
                # 부족한 프레임은 마지막 프레임으로 패딩
                last_frame = sequence[-1] if sequence else [0] * len(sequence[0]) if sequence else [0] * 185
                while len(sequence) < MAX_SEQUENCE_LENGTH:
                    sequence.append(last_frame.copy())
            
            # numpy 배열로 변환
            sequence_array = np.array(sequence, dtype=np.float32)
            
            # 차원 확인 및 조정
            if len(sequence_array.shape) == 2:
                # (frames, features) -> (1, frames, features)
                sequence_array = sequence_array.reshape(1, -1, sequence_array.shape[1])
            
            return sequence_array
            
        except Exception as e:
            print(f"❌ 데이터 전처리 오류: {e}")
            raise
    
    def predict_motion(self, input_data: Dict) -> Dict:
        """
        Unity에서 전송된 데이터로 동작을 예측
        
        Args:
            input_data: Unity에서 전송된 JSON 데이터
            
        Returns:
            예측 결과 딕셔너리
        """
        try:
            # 입력 데이터 전처리
            X = self.preprocess_input_data(input_data)
            
            # 모델 예측
            prediction = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            
            # 결과 해석
            predicted_label = int(prediction[0])
            predicted_motion = self.label_map[predicted_label]
            confidence_scores = probabilities[0].tolist()
            
            # 결과 반환
            result = {
                "predicted_motion": predicted_motion,
                "predicted_label": predicted_label,
                "confidence_scores": confidence_scores,
                "confidence_per_motion": {
                    "Hold": confidence_scores[1],  # Hold는 라벨 1
                    "Pick": confidence_scores[0],  # Pick은 라벨 0
                    "Place": confidence_scores[2]  # Place는 라벨 2
                },
                "input_features_shape": X.shape,
                "processing_info": {
                    "use_enhanced_features": self.use_enhanced_features,
                    "n_frames": X.shape[1],
                    "n_features": X.shape[2]
                }
            }
            
            return result
            
        except Exception as e:
            print(f"❌ 예측 오류: {e}")
            return {
                "error": str(e),
                "predicted_motion": "Unknown",
                "confidence_scores": [0, 0, 0]
            }
    
    def test_with_sample_data(self):
        """샘플 데이터로 모델 테스트"""
        print("\n🧪 샘플 데이터로 모델 테스트...")
        
        # 테스트용 샘플 데이터 (Gaze_Hold_010.json 형식)
        sample_data = {
            "sessionId": "test-session",
            "startTime": "2025-01-01 00:00:00",
            "interactionTargetObject": "TestObject",
            "interactionTargetPosition": {"x": 0.1, "y": 0.2, "z": 0.3},
            "frames": [
                {
                    "timestamp": 0.0,
                    "isRightHand": True,
                    "joints": [
                        {
                            "jointName": "Wrist",
                            "position": {"x": 0.1, "y": 0.1, "z": 0.1},
                            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                            "confidence": 1.0
                        }
                    ]
                }
            ]
        }
        
        try:
            result = self.predict_motion(sample_data)
            print(f"✅ 테스트 성공!")
            print(f"   예측된 동작: {result['predicted_motion']}")
            print(f"   신뢰도: {result['confidence_per_motion']}")
            print(f"   입력 특징 형태: {result['input_features_shape']}")
            
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")

# 메인 실행 함수
def main():
    """메인 실행 함수"""
    print("🚀 Unity Python Bridge for Motion Classification")
    print("=" * 60)
    
    try:
        # 브리지 초기화
        bridge = UnityPythonBridge(use_enhanced_features=True)
        
        # 샘플 데이터로 테스트
        bridge.test_with_sample_data()
        
        print("\n🎯 브리지가 성공적으로 초기화되었습니다!")
        print("Unity에서 JSON 데이터를 전송하면 동작을 분류할 수 있습니다.")
        
    except Exception as e:
        print(f"❌ 브리지 초기화 실패: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
