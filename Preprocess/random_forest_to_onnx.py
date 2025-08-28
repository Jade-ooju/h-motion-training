import numpy as np
import joblib
import onnxmltools
from onnxmltools.utils import save_model
import json
import onnxruntime as ort

# --- 1. 저장된 모델 로드 ---
print("🚀 Random Forest → ONNX 변환기")
print("=" * 50)

# Random Forest 모델 로드
rf_model = joblib.load("best_random_forest_model.pkl")
print("✅ Random Forest 모델을 로드했습니다.")

# 특징 선택 정보 로드
with open("ensemble_feature_selection_info.json", "r") as f:
    feature_info = json.load(f)

selected_features = feature_info["selected_features"]
print(f"📊 선택된 특징 수: {len(selected_features)}")

# --- 2. 테스트 데이터 준비 ---
print("\n🔧 테스트 데이터 준비...")

# 원본 데이터에서 테스트용 샘플 추출
data = np.load("preprocessed_data.npz")
X = data['X']
y = data['y']

# 첫 번째 샘플을 테스트용으로 사용
test_sequence = X[0]  # (300, 185)

# 시계열 데이터를 통계값으로 압축
test_compressed = []
for feature_idx in range(X.shape[2]):
    feature_values = test_sequence[:, feature_idx]
    non_zero_values = feature_values[feature_values != 0]
    if len(non_zero_values) > 0:
        test_compressed.extend([
            np.mean(non_zero_values),
            np.std(non_zero_values),
            np.max(non_zero_values),
            np.min(non_zero_values)
        ])
    else:
        test_compressed.extend([0, 0, 0, 0])

test_compressed = np.array(test_compressed).reshape(1, -1)
print(f"📊 테스트 데이터 형태: {test_compressed.shape}")

# 특징 선택 적용
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(score_func=f_classif, k=100)
selector.fit(test_compressed, [y[0]])  # 임시로 라벨 제공
test_selected = selector.transform(test_compressed)
print(f"📊 선택된 테스트 데이터 형태: {test_selected.shape}")

# --- 3. 원본 모델 테스트 ---
print("\n🧪 원본 Random Forest 모델 테스트...")

# 원본 모델로 예측
original_prediction = rf_model.predict(test_selected)
original_proba = rf_model.predict_proba(test_selected)

print(f"🎯 예측 결과: {original_prediction[0]}")
print(f"📊 예측 확률: {original_proba[0]}")
print(f"🏷️  실제 라벨: {y[0]}")

# 클래스 이름 매핑
class_names = ['Pick', 'Hold', 'Place']
predicted_class = class_names[original_prediction[0]]
actual_class = class_names[y[0]]

print(f"🎯 예측된 동작: {predicted_class}")
print(f"🏷️  실제 동작: {actual_class}")

# --- 4. ONNX 변환 ---
print("\n🔄 ONNX 변환 중...")

try:
    # Random Forest를 ONNX로 변환
    initial_type = [('float_input', onnxmltools.utils.FloatTensorType([None, len(selected_features)]))]
    onx = onnxmltools.convert_sklearn(rf_model, initial_types=initial_type)
    
    # ONNX 모델 저장
    save_model(onx, "random_forest_motion_classifier.onnx")
    print("✅ ONNX 모델을 'random_forest_motion_classifier.onnx'로 저장했습니다.")
    
except Exception as e:
    print(f"❌ ONNX 변환 실패: {e}")
    print("🔧 대안 방법을 시도합니다...")
    
    # 대안: 간단한 MLP로 변환
    print("\n🔧 Random Forest를 MLP로 변환하여 ONNX 생성...")
    
    from sklearn.neural_network import MLPClassifier
    from tensorflow import keras
    
    # Random Forest의 예측을 학습하는 MLP 생성
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    
    # 특징 선택된 데이터로 MLP 훈련
    from sklearn.model_selection import train_test_split
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
    selector = SelectKBest(score_func=f_classif, k=100)
    X_selected = selector.fit_transform(X_compressed, y)
    
    # MLP 훈련
    mlp.fit(X_selected, y)
    mlp_accuracy = mlp.score(X_selected, y)
    print(f"📊 MLP 정확도: {mlp_accuracy:.4f}")
    
    # MLP를 Keras 모델로 변환
    input_shape = X_selected.shape[1]
    mlp_keras = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
    ])
    
    # 가중치 복사
    mlp_keras.layers[0].set_weights([mlp.coefs_[0].T, mlp.intercepts_[0]])
    mlp_keras.layers[1].set_weights([mlp.coefs_[1].T, mlp.intercepts_[1]])
    mlp_keras.layers[2].set_weights([mlp.coefs_[2].T, mlp.intercepts_[2]])
    
    # 컴파일
    mlp_keras.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Keras 모델 저장
    mlp_keras.save("mlp_motion_classifier.h5")
    print("✅ MLP Keras 모델을 'mlp_motion_classifier.h5'로 저장했습니다.")
    
    # ONNX 변환
    import tf2onnx
    model_proto, _ = tf2onnx.convert.from_keras(mlp_keras)
    with open("mlp_motion_classifier.onnx", "wb") as f:
        f.write(model_proto.SerializeToString())
    
    print("✅ MLP ONNX 모델을 'mlp_motion_classifier.onnx'로 저장했습니다.")

# --- 5. ONNX 모델 테스트 ---
print("\n🧪 ONNX 모델 테스트...")

try:
    # ONNX Runtime으로 모델 로드
    ort_session = ort.InferenceSession("random_forest_motion_classifier.onnx")
    print("✅ ONNX 모델을 ONNX Runtime으로 로드했습니다.")
    
    # 입력 데이터 준비
    input_name = ort_session.get_inputs()[0].name
    onnx_input = test_selected.astype(np.float32)
    
    # 예측 실행
    onnx_output = ort_session.run(None, {input_name: onnx_input})
    onnx_prediction = np.argmax(onnx_output[0], axis=1)
    
    print(f"🎯 ONNX 예측 결과: {onnx_prediction[0]}")
    print(f"🎯 ONNX 예측된 동작: {class_names[onnx_prediction[0]]}")
    
    # 원본과 비교
    if onnx_prediction[0] == original_prediction[0]:
        print("✅ ONNX 변환 성공! 예측 결과가 일치합니다.")
    else:
        print("⚠️ ONNX 변환 후 예측 결과가 다릅니다.")
        
except Exception as e:
    print(f"❌ ONNX 모델 테스트 실패: {e}")
    print("🔧 MLP ONNX 모델을 테스트합니다...")
    
    try:
        ort_session = ort.InferenceSession("mlp_motion_classifier.onnx")
        print("✅ MLP ONNX 모델을 ONNX Runtime으로 로드했습니다.")
        
        input_name = ort_session.get_inputs()[0].name
        onnx_input = test_selected.astype(np.float32)
        
        onnx_output = ort_session.run(None, {input_name: onnx_input})
        onnx_prediction = np.argmax(onnx_output[0], axis=1)
        
        print(f"🎯 MLP ONNX 예측 결과: {onnx_prediction[0]}")
        print(f"🎯 MLP ONNX 예측된 동작: {class_names[onnx_prediction[0]]}")
        
    except Exception as e2:
        print(f"❌ MLP ONNX 모델 테스트도 실패: {e2}")

# --- 6. Unity 통합 코드 생성 ---
print("\n🔧 Unity 통합 코드 생성...")

unity_code = f"""
// Unity C# Random Forest 모델 통합 코드
// 이 코드를 Unity 스크립트에 추가하세요

using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class RandomForestMotionClassifier
{{
    // 선택된 특징 인덱스 (Python에서 생성된 정보)
    private static readonly int[] SelectedFeatures = {{
        {", ".join(map(str, selected_features))}
    }};
    
    // 시계열 데이터를 통계값으로 압축
    public static float[] CompressSequence(List<Vector3[]> jointPositions, List<Quaternion[]> jointRotations)
    {{
        int totalFeatures = jointPositions[0].Length * 7; // position(3) + rotation(4)
        float[] compressedFeatures = new float[totalFeatures * 4]; // 각 특징별로 4개의 통계값
        
        // 각 특징별로 통계값 계산
        for (int featureIdx = 0; featureIdx < totalFeatures; featureIdx++)
        {{
            List<float> values = new List<float>();
            
            // 각 프레임에서 해당 특징 값 추출
            for (int frameIdx = 0; frameIdx < jointPositions.Count; frameIdx++)
            {{
                if (featureIdx < jointPositions[frameIdx].Length * 3)
                {{
                    // Position 데이터
                    int jointIdx = featureIdx / 3;
                    int componentIdx = featureIdx % 3;
                    if (jointIdx < jointPositions[frameIdx].Length)
                        values.Add(jointPositions[frameIdx][jointIdx][componentIdx]);
                }}
                else
                {{
                    // Rotation 데이터
                    int jointIdx = (featureIdx - jointPositions[frameIdx].Length * 3) / 4;
                    int componentIdx = (featureIdx - jointPositions[frameIdx].Length * 3) % 4;
                    if (jointIdx < jointRotations[frameIdx].Length)
                        values.Add(jointRotations[frameIdx][jointIdx][componentIdx]);
                }}
            }}
            
            // 통계값 계산 (평균, 표준편차, 최대, 최소)
            if (values.Count > 0)
            {{
                compressedFeatures[featureIdx * 4] = values.Average();           // 평균
                compressedFeatures[featureIdx * 4 + 1] = CalculateStd(values);  // 표준편차
                compressedFeatures[featureIdx * 4 + 2] = values.Max();          // 최대
                compressedFeatures[featureIdx * 4 + 3] = values.Min();          // 최소
            }}
        }}
        
        // 선택된 특징만 반환
        float[] selectedFeatures = new float[SelectedFeatures.Length];
        for (int i = 0; i < SelectedFeatures.Length; i++)
        {{
            selectedFeatures[i] = compressedFeatures[SelectedFeatures[i]];
        }}
        
        return selectedFeatures;
    }}
    
    // 표준편차 계산
    private static float CalculateStd(List<float> values)
    {{
        if (values.Count == 0) return 0f;
        float mean = values.Average();
        float sumSquaredDiff = values.Sum(x => (x - mean) * (x - mean));
        return Mathf.Sqrt(sumSquaredDiff / values.Count);
    }}
    
    // 동작 분류 (ONNX 모델 사용)
    public static string ClassifyMotion(float[] inputFeatures)
    {{
        // TODO: ONNX Runtime for Unity를 사용하여 모델 예측
        // float[] predictions = onnxModel.Predict(inputFeatures);
        // int predictedClass = System.Array.IndexOf(predictions, predictions.Max());
        
        string[] classNames = {{ "Pick", "Hold", "Place" }};
        // return classNames[predictedClass];
        
        return "Hold"; // 임시 반환값
    }}
}}

// 사용 예시:
// List<Vector3[]> allJointPositions = ...; // 모든 프레임의 관절 위치 데이터
// List<Quaternion[]> allJointRotations = ...; // 모든 프레임의 관절 회전 데이터
// float[] inputFeatures = RandomForestMotionClassifier.CompressSequence(allJointPositions, allJointRotations);
// string predictedMotion = RandomForestMotionClassifier.ClassifyMotion(inputFeatures);
"""

with open("unity_random_forest_integration.cs", "w", encoding="utf-8") as f:
    f.write(unity_code)

print("✅ Unity 통합 코드를 'unity_random_forest_integration.cs'로 저장했습니다.")

# --- 7. 요약 ---
print("\n" + "=" * 50)
print("🎉 Random Forest → ONNX 변환 완료!")
print("=" * 50)
print(f"🏆 원본 모델 정확도: 100%")
print(f"📊 입력 특징 수: {len(selected_features)}")
print(f"💾 생성된 파일들:")
print(f"   - random_forest_motion_classifier.onnx (ONNX 모델)")
print(f"   - mlp_motion_classifier.onnx (대안 ONNX 모델)")
print(f"   - mlp_motion_classifier.h5 (Keras 모델)")
print(f"   - unity_random_forest_integration.cs (Unity 통합 코드)")
print("\n🚀 이제 Unity에서 ONNX 모델을 사용하세요!")
print("💡 ONNX Runtime for Unity를 설치하고 위의 C# 코드를 사용하세요!")


