#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unity Python Bridge for Motion Classification
Unity에서 전송된 관절 데이터를 받아 동작을 분류하는 Python 브리지
"""

import json
import numpy as np
import joblib
import sys
import os
from typing import Dict, List, Tuple, Optional

class UnityPythonBridge:
    """Unity와 통신하여 동작을 분류하는 Python 브리지"""
    
    def __init__(self, model_path: str = "../Models/best_random_forest_model.pkl", 
                 feature_info_path: str = "../Models/ensemble_features.json"):
        """
        Python 브리지 초기화
        
        Args:
            model_path: 훈련된 Random Forest 모델 파일 경로
            feature_info_path: 특징 선택 정보 파일 경로
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
            
            # 라벨 매핑
            self.label_map = {0: "Hold", 1: "Pick", 2: "Place"}
            self.reverse_label_map = {v: k for k, v in self.label_map.items()}
            
            print(f"🎯 동작 분류 준비 완료!")
            print(f"   - 모델: {type(self.model).__name__}")
            print(f"   - 특징 수: {self.feature_info['n_features']}")
            print(f"   - 선택된 특징: {len(self.feature_info['selected_features'])}")
            
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
        
        # Gaze Target Position (x, y, z) - 현재는 없으므로 0으로 채움
        # 향후 시선 데이터가 추가되면 여기서 처리
        flat_data.extend([0, 0, 0])
        
        return flat_data
    
    def compress_sequence(self, sequence: List[List[float]]) -> np.ndarray:
        """
        시계열 데이터를 통계값으로 압축
        
        Args:
            sequence: 시계열 데이터 (프레임들의 리스트)
            
        Returns:
            압축된 특징 벡터
        """
        if not sequence:
            return np.zeros(self.feature_info['n_features'])
        
        # numpy 배열로 변환
        data = np.array(sequence)
        
        # 각 특징에 대해 통계값 계산
        compressed_features = []
        
        for i in range(data.shape[1]):
            feature_values = data[:, i]
            compressed_features.extend([
                np.mean(feature_values),    # 평균
                np.std(feature_values),     # 표준편차
                np.max(feature_values),     # 최대값
                np.min(feature_values)      # 최소값
            ])
        
        return np.array(compressed_features)
    
    def select_features(self, compressed_data: np.ndarray) -> np.ndarray:
        """
        특징 선택을 통해 최적의 특징만 선택
        
        Args:
            compressed_data: 압축된 특징 벡터
            
        Returns:
            선택된 특징만 포함된 벡터
        """
        selected_indices = self.feature_info['selected_features']
        return compressed_data[selected_indices]
    
    def predict_motion(self, sequence_data: List[Dict]) -> Tuple[str, np.ndarray, float]:
        """
        시계열 관절 데이터로부터 동작을 예측
        
        Args:
            sequence_data: 시계열 관절 데이터 (프레임들의 리스트)
            
        Returns:
            (예측된 동작, 확률 분포, 신뢰도)
        """
        try:
            # 1. 각 프레임을 평탄화
            flattened_frames = [self.flatten_frame(frame) for frame in sequence_data]
            
            # 2. 시계열 데이터를 통계값으로 압축
            compressed_features = self.compress_sequence(flattened_frames)
            
            # 3. 특징 선택
            selected_features = self.select_features(compressed_features)
            
            # 4. 모델 예측
            prediction = self.model.predict([selected_features])[0]
            probabilities = self.model.predict_proba([selected_features])[0]
            
            # 5. 결과 변환
            predicted_motion = self.label_map[prediction]
            confidence = np.max(probabilities)
            
            return predicted_motion, probabilities, confidence
            
        except Exception as e:
            print(f"❌ 예측 오류: {e}")
            return "Error", np.array([0, 0, 0]), 0.0
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        return {
            "model_type": type(self.model).__name__,
            "n_features": self.feature_info['n_features'],
            "n_selected_features": len(self.feature_info['selected_features']),
            "labels": list(self.label_map.values()),
            "feature_selection_method": self.feature_info.get('method', 'SelectKBest')
        }
    
    def test_prediction(self):
        """테스트 예측 수행"""
        print("\n🧪 테스트 예측 수행...")
        
        # 테스트 데이터 생성 (Hold 동작 시뮬레이션)
        test_sequence = []
        for i in range(30):  # 30 프레임
            frame = {
                "joints": [
                    {
                        "jointName": "Wrist",
                        "position": {"x": 0.1 + i*0.001, "y": 0.2, "z": 0.3},
                        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
                    },
                    {
                        "jointName": "ForearmWrist",
                        "position": {"x": 0.1 + i*0.001, "y": 0.2, "z": 0.3},
                        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
                    }
                    # ... 다른 관절들도 비슷하게 생성
                ]
            }
            test_sequence.append(frame)
        
        # 예측 수행
        motion, probs, conf = self.predict_motion(test_sequence)
        
        print(f"🎯 예측 결과:")
        print(f"   - 동작: {motion}")
        print(f"   - 확률: Hold={probs[0]:.2f}, Pick={probs[1]:.2f}, Place={probs[2]:.2f}")
        print(f"   - 신뢰도: {conf:.2f}")
        
        return motion, probs, conf

def main():
    """메인 함수 - Unity와의 통신 처리"""
    try:
        print("🚀 Unity Python Bridge 시작...")
        
        # 브리지 초기화
        bridge = UnityPythonBridge()
        
        # 테스트 예측 수행
        bridge.test_prediction()
        
        print("\n✅ Python Bridge 초기화 완료!")
        print("🔄 Unity에서 데이터를 전송하면 동작을 분류합니다...")
        
        # Unity와의 통신 대기 (실제 구현에서는 표준 입출력 또는 소켓 사용)
        while True:
            try:
                # Unity에서 전송된 데이터 읽기
                input_data = input().strip()
                
                if input_data.lower() == 'quit':
                    break
                
                # JSON 데이터 파싱
                data = json.loads(input_data)
                
                # 동작 예측
                motion, probs, conf = bridge.predict_motion(data['sequence'])
                
                # 결과를 Unity로 전송
                result = {
                    "motion": motion,
                    "probabilities": probs.tolist(),
                    "confidence": float(conf)
                }
                
                print(json.dumps(result))
                
            except json.JSONDecodeError:
                print(json.dumps({"error": "Invalid JSON format"}))
            except Exception as e:
                print(json.dumps({"error": str(e)}))
                
    except KeyboardInterrupt:
        print("\n👋 Python Bridge 종료")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
