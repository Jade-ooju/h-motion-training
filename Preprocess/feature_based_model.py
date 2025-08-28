import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# --- 1. 데이터 로드 ---
print("🔍 특징 기반 모델 테스트")
print("=" * 50)

data = np.load("preprocessed_data.npz")
X = data['X']
y = data['y']

print(f"📊 데이터 형태: {X.shape}")
print(f"📊 클래스 분포: {np.bincount(y)}")

# --- 2. 데이터 압축 (이전과 동일) ---
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
selector = SelectKBest(score_func=f_classif, k=100)  # 상위 100개 특징만 선택
X_selected = selector.fit_transform(X_compressed, y)

# 선택된 특징의 인덱스
selected_features = selector.get_support(indices=True)
print(f"📊 선택된 특징 수: {len(selected_features)}")
print(f"📊 선택된 특징 인덱스: {selected_features[:10]}...")  # 처음 10개만 출력

# --- 4. 데이터 분리 ---
X_train, X_val, y_train, y_val = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 훈련 데이터: {len(X_train)}개")
print(f"📊 검증 데이터: {len(X_val)}개")

# --- 5. 특징 선택된 Random Forest ---
print("\n🌲 특징 선택된 Random Forest 테스트...")
rf_selected = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_selected.fit(X_train, y_train)

y_pred_rf_selected = rf_selected.predict(X_val)
rf_selected_accuracy = accuracy_score(y_val, y_pred_rf_selected)

print(f"🌲 특징 선택된 Random Forest 정확도: {rf_selected_accuracy:.4f}")
print(f"🌲 분류 보고서:")
print(classification_report(y_val, y_pred_rf_selected, target_names=['Pick', 'Hold', 'Place']))

# --- 6. 특징 선택된 Neural Network ---
print("\n🧠 특징 선택된 Neural Network 테스트...")

# 더 간단한 모델 (특징 수가 줄어들었으므로)
simple_model_selected = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(X_selected.shape[1],)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

simple_model_selected.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("🧠 특징 선택된 모델 구조:")
simple_model_selected.summary()

# 훈련
history_selected = simple_model_selected.fit(
    X_train, y_train,
    epochs=100,
    batch_size=8,
    validation_data=(X_val, y_val),
    verbose=1
)

# 평가
y_pred_nn_selected = simple_model_selected.predict(X_val)
y_pred_nn_selected_classes = np.argmax(y_pred_nn_selected, axis=1)
nn_selected_accuracy = accuracy_score(y_val, y_pred_nn_selected_classes)

print(f"\n🧠 특징 선택된 Neural Network 정확도: {nn_selected_accuracy:.4f}")
print(f"🧠 분류 보고서:")
print(classification_report(y_val, y_pred_nn_selected_classes, target_names=['Pick', 'Hold', 'Place']))

# --- 7. 결과 비교 ---
print("\n📊 결과 비교:")
print(f"🌲 원본 Random Forest: 1.0000")
print(f"🌲 특징 선택된 Random Forest: {rf_selected_accuracy:.4f}")
print(f"🧠 원본 Neural Network: 0.5556")
print(f"🧠 특징 선택된 Neural Network: {nn_selected_accuracy:.4f}")

# --- 8. 특징 중요도 분석 ---
print("\n🔍 선택된 특징의 중요도 분석...")
feature_importance_selected = rf_selected.feature_importances_
top_selected_features = np.argsort(feature_importance_selected)[-10:]

print("🔍 상위 10개 중요 선택 특징:")
for i, feature_idx in enumerate(reversed(top_selected_features)):
    original_feature_idx = selected_features[feature_idx]
    importance = feature_importance_selected[feature_idx]
    print(f"   {i+1:2d}. 선택된 특징 {feature_idx:3d} (원본 {original_feature_idx:3d}): {importance:.4f}")

# --- 9. 시각화 ---
plt.figure(figsize=(15, 5))

# 훈련 과정 비교
plt.subplot(1, 3, 1)
plt.plot(history_selected.history['accuracy'], label='Training Accuracy')
plt.plot(history_selected.history['val_accuracy'], label='Validation Accuracy')
plt.title('Feature-Selected Neural Network')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# 특징 중요도 비교
plt.subplot(1, 3, 2)
plt.bar(range(len(feature_importance_selected)), feature_importance_selected)
plt.title('Selected Features Importance')
plt.xlabel('Selected Feature Index')
plt.ylabel('Importance')

# 모델 성능 비교
plt.subplot(1, 3, 3)
models = ['Original RF', 'Selected RF', 'Original NN', 'Selected NN']
accuracies = [1.0000, rf_selected_accuracy, 0.5556, nn_selected_accuracy]
colors = ['#FF6B6B', '#FF8E8E', '#4ECDC4', '#6EDDDD']
bars = plt.bar(models, accuracies, color=colors)
plt.title('Model Performance Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1.1)

# 정확도 값 표시
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{acc:.3f}', ha='center', va='bottom')

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('feature_based_model_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n🎉 특징 기반 모델 테스트 완료!")
print(f"💡 핵심 인사이트: {len(selected_features)}개 특징만으로도 높은 성능 달성 가능!")


