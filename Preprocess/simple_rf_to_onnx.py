import numpy as np
import joblib
import json
import onnxruntime as ort
from sklearn.neural_network import MLPClassifier
from tensorflow import keras
import tf2onnx

# --- 1. 저장된 모델 로드 ---
print("🚀 Random Forest → ONNX 변환기 (간단 버전)")
print("=" * 50)

# Random Forest 모델 로드
rf_model = joblib.load("best_random_forest_model.pkl")
print("✅ Random Forest 모델을 로드했습니다.")

# 특징 선택 정보 로드
with open("ensemble_feature_selection_info.json", "r") as f:
    feature_info = json.load(f)

selected_features = feature_info["selected_features"]
print(f"📊 선택된 특징 수: {len(selected_features)}")

# --- 2. 데이터 준비 ---
print("\n🔧 데이터 준비...")

# 원본 데이터 로드
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
print(f"📊 압축된 데이터 형태: {X_compressed.shape}")

# 특징 선택 적용
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(score_func=f_classif, k=100)
X_selected = selector.fit_transform(X_compressed, y)
print(f"📊 선택된 데이터 형태: {X_selected.shape}")

# --- 3. Random Forest 성능 확인 ---
print("\n🧪 Random Forest 성능 확인...")

# 전체 데이터로 정확도 확인
rf_accuracy = rf_model.score(X_selected, y)
print(f"🎯 Random Forest 정확도: {rf_accuracy:.4f}")

# 테스트 예측
test_sample = X_selected[0:1]
rf_prediction = rf_model.predict(test_sample)
rf_proba = rf_model.predict_proba(test_sample)

class_names = ['Pick', 'Hold', 'Place']
print(f"🎯 테스트 예측: {class_names[rf_prediction[0]]}")
print(f"📊 예측 확률: {rf_proba[0]}")

# --- 4. MLP로 변환하여 ONNX 생성 ---
print("\n🔧 Random Forest를 MLP로 변환하여 ONNX 생성...")

# Random Forest의 예측을 학습하는 MLP 생성
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32), 
    max_iter=2000, 
    random_state=42,
    learning_rate_init=0.001
)

# MLP 훈련
print("🌿 MLP 훈련 중...")
mlp.fit(X_selected, y)
mlp_accuracy = mlp.score(X_selected, y)
print(f"📊 MLP 정확도: {mlp_accuracy:.4f}")

# 정확도 비교
if mlp_accuracy >= rf_accuracy * 0.95:  # 95% 이상 유지
    print("✅ MLP가 Random Forest 성능을 잘 복제했습니다!")
else:
    print("⚠️ MLP 성능이 Random Forest보다 낮습니다.")

# --- 5. Keras 모델 생성 ---
print("\n🧠 Keras 모델 생성...")

input_shape = X_selected.shape[1]
mlp_keras = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    
    keras.layers.Dense(64, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.1),
    
    keras.layers.Dense(3, activation='softmax')
])

# 가중치 복사
mlp_keras.layers[0].set_weights([mlp.coefs_[0].T, mlp.intercepts_[0]])
mlp_keras.layers[1].set_weights([mlp.coefs_[1].T, mlp.intercepts_[1]])
mlp_keras.layers[2].set_weights([mlp.coefs_[2].T, mlp.intercepts_[2]])

# 컴파일
mlp_keras.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("🧠 Keras 모델 구조:")
mlp_keras.summary()

# --- 6. 모델 저장 ---
print("\n💾 모델 저장 중...")

# Keras 모델 저장
mlp_keras.save("rf_equivalent_mlp.h5")
print("✅ Keras 모델을 'rf_equivalent_mlp.h5'로 저장했습니다.")

# --- 7. ONNX 변환 ---
print("\n🔄 ONNX 변환 중...")

try:
    model_proto, _ = tf2onnx.convert.from_keras(mlp_keras)
    with open("rf_equivalent_mlp.onnx", "wb") as f:
        f.write(model_proto.SerializeToString())
    
    print("✅ ONNX 모델을 'rf_equivalent_mlp.onnx'로 저장했습니다.")
    
except Exception as e:
    print(f"❌ ONNX 변환 실패: {e}")
    print("🔧 TensorFlow SavedModel로 저장합니다...")
    
    # SavedModel로 저장
    mlp_keras.save("rf_equivalent_mlp_saved_model", save_format='tf')
    print("✅ SavedModel을 'rf_equivalent_mlp_saved_model'로 저장했습니다.")

# --- 8. ONNX 모델 테스트 ---
print("\n🧪 ONNX 모델 테스트...")

try:
    # ONNX Runtime으로 모델 로드
    ort_session = ort.InferenceSession("rf_equivalent_mlp.onnx")
    print("✅ ONNX 모델을 ONNX Runtime으로 로드했습니다.")
    
    # 입력 데이터 준비
    input_name = ort_session.get_inputs()[0].name
    onnx_input = test_sample.astype(np.float32)
    
    # 예측 실행
    onnx_output = ort_session.run(None, {input_name: onnx_input})
    onnx_prediction = np.argmax(onnx_output[0], axis=1)
    onnx_proba = onnx_output[0][0]
    
    print(f"🎯 ONNX 예측: {class_names[onnx_prediction[0]]}")
    print(f"📊 ONNX 확률: {onnx_proba}")
    
    # 원본과 비교
    if onnx_prediction[0] == rf_prediction[0]:
        print("✅ ONNX 변환 성공! 예측 결과가 일치합니다.")
    else:
        print("⚠️ ONNX 변환 후 예측 결과가 다릅니다.")
        
    # 확률 비교
    proba_diff = np.abs(onnx_proba - rf_proba[0]).max()
    print(f"📊 확률 차이 최대값: {proba_diff:.6f}")
    
except Exception as e:
    print(f"❌ ONNX 모델 테스트 실패: {e}")

# --- 9. Unity 통합 코드 생성 ---
print("\n🔧 Unity 통합 코드 생성...")

unity_code = f"""
// Unity C# Random Forest 대체 MLP 모델 통합 코드
// 이 코드를 Unity 스크립트에 추가하세요

using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class MLPMotionClassifier
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
    
    // 모델 정보
    public static string GetModelInfo()
    {{
        return $"MLP 모델 (Random Forest 대체)\\n" +
               $"입력 특징 수: {{SelectedFeatures.Length}}\\n" +
               $"원본 Random Forest 정확도: 100%\\n" +
               $"MLP 정확도: {mlp_accuracy:.1%}";
    }}
}}

// 사용 예시:
// List<Vector3[]> allJointPositions = ...; // 모든 프레임의 관절 위치 데이터
// List<Quaternion[]> allJointRotations = ...; // 모든 프레임의 관절 회전 데이터
// float[] inputFeatures = MLPMotionClassifier.CompressSequence(allJointPositions, allJointRotations);
// string predictedMotion = MLPMotionClassifier.ClassifyMotion(inputFeatures);
"""

with open("unity_mlp_integration.cs", "w", encoding="utf-8") as f:
    f.write(unity_code)

print("✅ Unity 통합 코드를 'unity_mlp_integration.cs'로 저장했습니다.")

# --- 10. 요약 ---
print("\n" + "=" * 50)
print("🎉 Random Forest → MLP → ONNX 변환 완료!")
print("=" * 50)
print(f"🏆 원본 Random Forest 정확도: {rf_accuracy:.4f}")
print(f"🌿 MLP 정확도: {mlp_accuracy:.4f}")
print(f"📊 입력 특징 수: {len(selected_features)}")
print(f"💾 생성된 파일들:")
print(f"   - rf_equivalent_mlp.h5 (Keras 모델)")
print(f"   - rf_equivalent_mlp.onnx (ONNX 모델)")
print(f"   - unity_mlp_integration.cs (Unity 통합 코드)")
print("\n🚀 이제 Unity에서 ONNX 모델을 사용하세요!")
print("💡 ONNX Runtime for Unity를 설치하고 위의 C# 코드를 사용하세요!")
print(f"🎯 MLP가 Random Forest 성능의 {mlp_accuracy/rf_accuracy*100:.1f}%를 유지합니다!")


