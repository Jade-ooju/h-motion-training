import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# --- 1. 데이터 로드 ---
print("🔍 간단한 모델로 데이터 학습 가능성 테스트")
print("=" * 50)

data = np.load("preprocessed_data.npz")
X = data['X']
y = data['y']

print(f"📊 데이터 형태: {X.shape}")
print(f"📊 클래스 분포: {np.bincount(y)}")

# --- 2. 데이터 전처리 ---
# 시계열 데이터를 평균, 표준편차, 최대, 최소값으로 압축
print("\n🔄 시계열 데이터를 통계값으로 압축...")

# 각 시퀀스의 통계값 계산
X_compressed = []
for sequence in X:
    # 각 특징별로 통계값 계산
    stats = []
    for feature_idx in range(X.shape[2]):
        feature_values = sequence[:, feature_idx]
        # 0이 아닌 값들만 사용 (패딩 제외)
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

# --- 3. 데이터 분리 ---
X_train, X_val, y_train, y_val = train_test_split(
    X_compressed, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 훈련 데이터: {len(X_train)}개")
print(f"📊 검증 데이터: {len(X_val)}개")

# --- 4. Random Forest로 테스트 ---
print("\n🌲 Random Forest 모델 테스트...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 예측
y_pred_rf = rf_model.predict(X_val)
rf_accuracy = accuracy_score(y_val, y_pred_rf)

print(f"🌲 Random Forest 정확도: {rf_accuracy:.4f}")
print(f"🌲 Random Forest 분류 보고서:")
print(classification_report(y_val, y_pred_rf, target_names=['Pick', 'Hold', 'Place']))

# --- 5. 간단한 Neural Network로 테스트 ---
print("\n🧠 간단한 Neural Network 테스트...")

# 간단한 MLP 모델
simple_model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_compressed.shape[1],)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

simple_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("🧠 간단한 모델 구조:")
simple_model.summary()

# 훈련
history = simple_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=8,
    validation_data=(X_val, y_val),
    verbose=1
)

# 평가
y_pred_nn = simple_model.predict(X_val)
y_pred_nn_classes = np.argmax(y_pred_nn, axis=1)
nn_accuracy = accuracy_score(y_val, y_pred_nn_classes)

print(f"\n🧠 Neural Network 정확도: {nn_accuracy:.4f}")
print(f"🧠 Neural Network 분류 보고서:")
print(classification_report(y_val, y_pred_nn_classes, target_names=['Pick', 'Hold', 'Place']))

# --- 6. 결과 비교 ---
print("\n📊 결과 비교:")
print(f"🌲 Random Forest: {rf_accuracy:.4f}")
print(f"🧠 Neural Network: {nn_accuracy:.4f}")

# --- 7. 특징 중요도 분석 (Random Forest) ---
print("\n🔍 특징 중요도 분석...")
feature_importance = rf_model.feature_importances_
top_features = np.argsort(feature_importance)[-10:]  # 상위 10개 특징

print("🔍 상위 10개 중요 특징:")
for i, feature_idx in enumerate(reversed(top_features)):
    print(f"   {i+1:2d}. 특징 {feature_idx:3d}: {feature_importance[feature_idx]:.4f}")

# --- 8. 시각화 ---
plt.figure(figsize=(15, 5))

# 훈련 과정
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Neural Network Training Process')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# 특징 중요도
plt.subplot(1, 3, 2)
plt.bar(range(len(feature_importance)), feature_importance)
plt.title('Random Forest Feature Importance')
plt.xlabel('Feature Index')
plt.ylabel('Importance')

# 모델 비교
plt.subplot(1, 3, 3)
models = ['Random Forest', 'Neural Network']
accuracies = [rf_accuracy, nn_accuracy]
colors = ['#FF6B6B', '#4ECDC4']
bars = plt.bar(models, accuracies, color=colors)
plt.title('Model Performance Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

# 정확도 값 표시
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{acc:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('simple_model_test_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n🎉 테스트 완료! 결과를 확인하세요.")
