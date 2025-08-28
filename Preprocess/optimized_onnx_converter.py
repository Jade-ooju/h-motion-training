import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
import tf2onnx
import json

# --- 1. 데이터 로드 ---
print("🚀 최적화된 ONNX 모델 생성기")
print("=" * 50)

data = np.load("preprocessed_data.npz")
X = data['X']
y = data['y']

print(f"📊 데이터 형태: {X.shape}")
print(f"📊 클래스 분포: {np.bincount(y)}")

# --- 2. 데이터 압축 ---
print("\n🔄 시계열 데이터를 통계값으로 압축...")

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

# --- 3. 특징 선택 ---
print("\n🔍 특징 선택을 통한 차원 축소...")

# F-test를 사용한 특징 선택
selector = SelectKBest(score_func=f_classif, k=100)
X_selected = selector.fit_transform(X_compressed, y)

# 선택된 특징의 인덱스
selected_features = selector.get_support(indices=True)
print(f"📊 선택된 특징 수: {len(selected_features)}")

# 특징 선택 정보 저장 (Unity에서 사용)
feature_selection_info = {
    "selected_features": selected_features.tolist(),
    "total_features": X_compressed.shape[1],
    "selected_count": len(selected_features),
    "compression_ratio": len(selected_features) / X_compressed.shape[1]
}

with open("feature_selection_info.json", "w") as f:
    json.dump(feature_selection_info, f, indent=2)

print(f"💾 특징 선택 정보를 'feature_selection_info.json'에 저장했습니다.")

# --- 4. 데이터 분리 ---
X_train, X_val, y_train, y_val = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 훈련 데이터: {len(X_train)}개")
print(f"📊 검증 데이터: {len(X_val)}개")

# --- 5. 최적화된 모델 생성 ---
print("\n🧠 최적화된 Neural Network 모델 생성...")

# 특징 수에 맞춘 최적화된 모델
optimized_model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_selected.shape[1],)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    
    keras.layers.Dense(32, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dropout(0.1),
    
    keras.layers.Dense(3, activation='softmax')
])

optimized_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("🧠 최적화된 모델 구조:")
optimized_model.summary()

# --- 6. 모델 훈련 ---
print("\n🚀 모델 훈련 시작...")

# Early Stopping과 Learning Rate Reduction
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=8,
    min_lr=1e-7,
    verbose=1
)

history = optimized_model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

print("🚀 모델 훈련 완료!")

# --- 7. 성능 평가 ---
print("\n📊 최종 성능 평가...")

# 검증 데이터로 평가
y_pred = optimized_model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)

from sklearn.metrics import accuracy_score, classification_report
final_accuracy = accuracy_score(y_val, y_pred_classes)

print(f"🎯 최종 검증 정확도: {final_accuracy:.4f}")
print(f"📋 분류 보고서:")
print(classification_report(y_val, y_pred_classes, target_names=['Pick', 'Hold', 'Place']))

# --- 8. 모델 저장 ---
print("\n💾 모델 저장 중...")

# Keras 모델 저장
optimized_model.save("optimized_motion_classifier.h5")
print("✅ Keras 모델을 'optimized_motion_classifier.h5'로 저장했습니다.")

# --- 9. ONNX 변환 ---
print("\n🔄 ONNX 변환 중...")

model_proto, _ = tf2onnx.convert.from_keras(optimized_model)
with open("optimized_motion_classifier.onnx", "wb") as f:
    f.write(model_proto.SerializeToString())

print("✅ ONNX 모델을 'optimized_motion_classifier.onnx'로 저장했습니다.")

# --- 10. Unity용 전처리 함수 생성 ---
print("\n🔧 Unity용 전처리 함수 생성...")

# C# 코드를 문자열로 생성 (f-string 사용하지 않음)
selected_features_str = ", ".join(map(str, selected_features))
unity_preprocessing_code = f"""// Unity C# 전처리 함수
// 이 코드를 Unity 스크립트에 추가하세요

using System.Collections.Generic;
using System.Linq;

public class MotionDataPreprocessor
{{
    // 선택된 특징 인덱스 (Python에서 생성된 정보)
    private static readonly int[] SelectedFeatures = {{
        {selected_features_str}
    }};
    
    // 시계열 데이터를 통계값으로 압축
    public static float[] CompressSequence(List<Vector3[]> jointPositions, List<Quaternion[]> jointRotations)
    {{
        int totalFeatures = jointPositions[0].Length * 7; // position(3) + rotation(4)
        float[] compressedFeatures = new float[totalFeatures];
        
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
                    values.Add(jointPositions[frameIdx][jointIdx][componentIdx]);
                }}
                else
                {{
                    // Rotation 데이터
                    int jointIdx = (featureIdx - jointPositions[frameIdx].Length * 3) / 4;
                    int componentIdx = (featureIdx - jointPositions[frameIdx].Length * 3) % 4;
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
        float mean = values.Average();
        float sumSquaredDiff = values.Sum(x => (x - mean) * (x - mean));
        return Mathf.Sqrt(sumSquaredDiff / values.Count);
    }}
}}

// 사용 예시:
// float[] inputFeatures = MotionDataPreprocessor.CompressSequence(jointPositions, jointRotations);
// float[] predictions = model.Predict(inputFeatures);
"""

with open("unity_preprocessing_code.cs", "w", encoding="utf-8") as f:
    f.write(unity_preprocessing_code)

print("✅ Unity용 전처리 코드를 'unity_preprocessing_code.cs'로 저장했습니다.")

# --- 11. 요약 정보 출력 ---
print("\n" + "=" * 50)
print("🎉 최적화된 ONNX 모델 생성 완료!")
print("=" * 50)
print(f"📊 원본 특징 수: {X.shape[2]}")
print(f"📊 압축된 특징 수: {X_compressed.shape[1]}")
print(f"📊 선택된 특징 수: {len(selected_features)}")
print(f"📊 차원 감소율: {(1 - len(selected_features) / X_compressed.shape[1]) * 100:.1f}%")
print(f"🎯 최종 정확도: {final_accuracy:.4f}")
print(f"💾 생성된 파일들:")
print(f"   - optimized_motion_classifier.h5 (Keras 모델)")
print(f"   - optimized_motion_classifier.onnx (Unity용 ONNX)")
print(f"   - feature_selection_info.json (특징 선택 정보)")
print(f"   - unity_preprocessing_code.cs (Unity 전처리 코드)")
print("\n🚀 이제 Unity에서 'optimized_motion_classifier.onnx'를 사용하세요!")
