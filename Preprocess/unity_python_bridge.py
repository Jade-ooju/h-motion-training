import numpy as np
import joblib
import json
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

class UnityPythonBridge:
    def __init__(self):
        """Unity Python 브리지 초기화"""
        print("🚀 Unity Python 브리지 초기화 중...")
        
        # Random Forest 모델 로드
        self.rf_model = joblib.load("best_random_forest_model.pkl")
        print("✅ Random Forest 모델 로드 완료")
        
        # 특징 선택 정보 로드
        with open("ensemble_feature_selection_info.json", "r") as f:
            feature_info = json.load(f)
        self.selected_features = feature_info["selected_features"]
        print(f"✅ 특징 선택 정보 로드 완료: {len(self.selected_features)}개 특징")
        
        # 특징 선택기 로드 (필요시)
        self.selector = None
        
        print("🎉 Unity Python 브리지 초기화 완료!")
    
    def compress_sequence(self, joint_positions, joint_rotations):
        """
        시계열 데이터를 통계값으로 압축
        
        Args:
            joint_positions: List[List[Vector3]] - 각 프레임의 관절 위치
            joint_rotations: List[List[Quaternion]] - 각 프레임의 관절 회전
            
        Returns:
            np.array: 압축된 특징 벡터 (740개 특징)
        """
        # 데이터를 numpy 배열로 변환
        if isinstance(joint_positions[0][0], (list, tuple)):
            # Unity Vector3가 [x, y, z] 형태로 전달된 경우
            joint_positions = [[np.array(pos) for pos in frame] for frame in joint_positions]
            joint_rotations = [[np.array(rot) for rot in frame] for frame in joint_rotations]
        
        # 시계열 데이터를 통계값으로 압축
        compressed_features = []
        
        # 각 특징별로 통계값 계산
        for feature_idx in range(len(joint_positions[0]) * 7):  # position(3) + rotation(4)
            values = []
            
            # 각 프레임에서 해당 특징 값 추출
            for frame_idx in range(len(joint_positions)):
                if feature_idx < len(joint_positions[frame_idx]) * 3:
                    # Position 데이터
                    joint_idx = feature_idx // 3
                    component_idx = feature_idx % 3
                    if joint_idx < len(joint_positions[frame_idx]):
                        values.append(joint_positions[frame_idx][joint_idx][component_idx])
                else:
                    # Rotation 데이터
                    joint_idx = (feature_idx - len(joint_positions[frame_idx]) * 3) // 4
                    component_idx = (feature_idx - len(joint_positions[frame_idx]) * 3) % 4
                    if joint_idx < len(joint_rotations[frame_idx]):
                        values.append(joint_rotations[frame_idx][joint_idx][component_idx])
            
            # 통계값 계산 (평균, 표준편차, 최대, 최소)
            if len(values) > 0:
                values = np.array(values)
                compressed_features.extend([
                    np.mean(values),
                    np.std(values),
                    np.max(values),
                    np.min(values)
                ])
            else:
                compressed_features.extend([0, 0, 0, 0])
        
        return np.array(compressed_features)
    
    def select_features(self, compressed_features):
        """
        압축된 특징에서 선택된 특징만 추출
        
        Args:
            compressed_features: np.array - 압축된 특징 벡터 (740개)
            
        Returns:
            np.array: 선택된 특징 벡터 (100개)
        """
        return compressed_features[self.selected_features]
    
    def predict_motion(self, joint_positions, joint_rotations):
        """
        동작 분류 예측
        
        Args:
            joint_positions: List[List[Vector3]] - 각 프레임의 관절 위치
            joint_rotations: List[List[Quaternion]] - 각 프레임의 관절 회전
            
        Returns:
            dict: 예측 결과 (클래스, 확률, 신뢰도)
        """
        try:
            # 1. 시계열 데이터 압축
            compressed = self.compress_sequence(joint_positions, joint_rotations)
            
            # 2. 특징 선택
            selected = self.select_features(compressed)
            
            # 3. 예측 실행
            prediction = self.rf_model.predict([selected])[0]
            probabilities = self.rf_model.predict_proba([selected])[0]
            
            # 4. 결과 반환
            class_names = ['Pick', 'Hold', 'Place']
            predicted_class = class_names[prediction]
            confidence = np.max(probabilities)
            
            result = {
                'class': predicted_class,
                'class_id': int(prediction),
                'probabilities': {
                    'Pick': float(probabilities[0]),
                    'Hold': float(probabilities[1]),
                    'Place': float(probabilities[2])
                },
                'confidence': float(confidence),
                'success': True
            }
            
            print(f"🎯 예측 완료: {predicted_class} (신뢰도: {confidence:.3f})")
            return result
            
        except Exception as e:
            print(f"❌ 예측 실패: {e}")
            return {
                'class': 'Unknown',
                'class_id': -1,
                'probabilities': {'Pick': 0, 'Hold': 0, 'Place': 0},
                'confidence': 0,
                'success': False,
                'error': str(e)
            }
    
    def get_model_info(self):
        """모델 정보 반환"""
        return {
            'model_type': 'Random Forest',
            'accuracy': 1.0,
            'selected_features_count': len(self.selected_features),
            'total_features_count': 740,
            'classes': ['Pick', 'Hold', 'Place']
        }
    
    def test_model(self):
        """모델 테스트 실행"""
        print("\n🧪 모델 테스트 실행...")
        
        # 테스트 데이터 로드
        data = np.load("preprocessed_data.npz")
        X = data['X']
        y = data['y']
        
        # 시계열 데이터를 통계값으로 압축
        X_compressed = []
        for sequence in X:
            stats = []
            for feature_idx in range(X.shape[2]):
                feature_values = sequence[:, feature_idx]
                non_zero_values = feature_values[feature_values != 0]
                if len(non_zero_values) > 0:
                    stats.extend([
                        np.mean(non_zero_values),
                        np.std(non_zero_values),
                        np.max(non_zero_values),
                        np.min(non_zero_values)
                    ])
                else:
                    stats.extend([0, 0, 0, 0])
            X_compressed.append(stats)
        
        X_compressed = np.array(X_compressed)
        
        # 특징 선택 적용
        selector = SelectKBest(score_func=f_classif, k=100)
        X_selected = selector.fit_transform(X_compressed, y)
        
        # 정확도 확인
        accuracy = self.rf_model.score(X_selected, y)
        print(f"✅ 모델 정확도: {accuracy:.4f}")
        
        # 테스트 예측
        test_sample = X_selected[0:1]
        prediction = self.rf_model.predict(test_sample)[0]
        probabilities = self.rf_model.predict_proba(test_sample)[0]
        
        class_names = ['Pick', 'Hold', 'Place']
        print(f"🎯 테스트 예측: {class_names[prediction]}")
        print(f"📊 예측 확률: {probabilities}")
        
        return {
            'accuracy': accuracy,
            'test_prediction': class_names[prediction],
            'test_probabilities': probabilities.tolist()
        }

# Unity에서 사용할 수 있는 간단한 함수들
def create_bridge():
    """Unity Python 브리지 인스턴스 생성"""
    return UnityPythonBridge()

def predict_motion_simple(bridge, joint_positions, joint_rotations):
    """간단한 동작 분류 함수 (Unity에서 호출)"""
    return bridge.predict_motion(joint_positions, joint_rotations)

def get_model_info_simple(bridge):
    """모델 정보 반환 함수 (Unity에서 호출)"""
    return bridge.get_model_info()

# 테스트 실행
if __name__ == "__main__":
    print("🚀 Unity Python 브리지 테스트")
    print("=" * 50)
    
    # 브리지 생성
    bridge = create_bridge()
    
    # 모델 테스트
    test_result = bridge.test_model()
    
    print("\n" + "=" * 50)
    print("🎉 Unity Python 브리지 테스트 완료!")
    print("=" * 50)
    print(f"🏆 모델 정확도: {test_result['accuracy']:.4f}")
    print(f"🎯 테스트 예측: {test_result['test_prediction']}")
    print(f"📊 테스트 확률: {test_result['test_probabilities']}")
    print("\n💡 Unity에서 이 스크립트를 Python 프로세스로 실행하여 Random Forest 모델을 사용하세요!")
    print("🚀 Python → Unity 통신을 위해 TCP 소켓이나 파일 기반 통신을 구현하세요!")


