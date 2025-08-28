import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import tf2onnx

# --- 1. 데이터 로드 ---
# 이전 단계에서 생성한 npz 파일을 불러옵니다.
data = np.load("preprocessed_data.npz")
X = data['X']
y = data['y']

print("데이터 로드 완료.")
print(f"X 데이터 형태: {X.shape}") # (45, 300, 367)
print(f"y 라벨 형태: {y.shape}")   # (45,)

# --- 2. 훈련/검증 데이터 분리 ---
# 데이터를 훈련용과 검증용으로 나눕니다. 80%는 훈련에, 20%는 검증에 사용합니다.
# random_state를 고정하면 항상 동일한 방식으로 데이터가 나뉩니다.
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n데이터 분리 완료.")
print(f"훈련 데이터 개수: {len(X_train)}")
print(f"검증 데이터 개수: {len(X_val)}")

# --- 3. 모델 아키텍처 정의 ---
# LSTM을 사용한 시계열 분류 모델을 만듭니다.
num_classes = len(np.unique(y)) # 클래스 개수 (Pick, Hold, Place -> 3개)
sequence_length = X.shape[1]    # 시퀀스 길이 (300)
num_features = X.shape[2]       # 특징 개수 (367)

model = keras.Sequential([
    # Masking 레이어: 패딩값(0)을 계산에서 제외시켜 모델 성능을 높입니다.
    keras.layers.Masking(mask_value=0., input_shape=(sequence_length, num_features)),
    
    # 첫 번째 LSTM 레이어: 더 많은 유닛과 return_sequences=True
    keras.layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    keras.layers.BatchNormalization(),
    
    # 두 번째 LSTM 레이어: 시퀀스 정보를 압축
    keras.layers.LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
    keras.layers.BatchNormalization(),
    
    # Dense 레이어들: 점진적으로 차원을 줄임
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    
    # 출력 레이어: 클래스별 확률을 출력
    keras.layers.Dense(num_classes, activation='softmax')
])

# 모델의 구조를 요약해서 보여줍니다.
model.summary()

# --- 4. 모델 컴파일 ---
# 모델의 학습 방식(손실 함수, 최적화기)을 설정합니다.
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # 더 낮은 학습률
    loss='sparse_categorical_crossentropy', # 라벨이 정수 형태일 때 사용
    metrics=['accuracy']
)

# --- 5. 모델 훈련 ---
print("\n모델 훈련을 시작합니다...")

# Early Stopping과 Learning Rate Reduction 추가
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

history = model.fit(
    X_train,
    y_train,
    epochs=100,  # 더 많은 에포크 (Early Stopping으로 자동 중단)
    batch_size=16, # 배치 크기 증가 (8 → 16)
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)
print("모델 훈련 완료!")

# --- 훈련 결과 시각화 ---
import matplotlib.pyplot as plt

# 훈련 과정 시각화
plt.figure(figsize=(12, 4))

# 정확도 그래프
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='훈련 정확도')
plt.plot(history.history['val_accuracy'], label='검증 정확도')
plt.title('모델 정확도')
plt.xlabel('에포크')
plt.ylabel('정확도')
plt.legend()
plt.grid(True)

# 손실 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='훈련 손실')
plt.plot(history.history['val_loss'], label='검증 손실')
plt.title('모델 손실')
plt.xlabel('에포크')
plt.ylabel('손실')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# 최종 성능 출력
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print(f"\n📊 최종 훈련 결과:")
print(f"   훈련 정확도: {final_train_acc:.4f}")
print(f"   검증 정확도: {final_val_acc:.4f}")
print(f"   훈련 손실: {final_train_loss:.4f}")
print(f"   검증 손실: {final_val_loss:.4f}")

# --- 6. 모델 저장 및 ONNX 변환 ---
# 훈련된 모델을 Keras 포맷으로 저장합니다.
model.save("motion_classifier.h5")
print("\n훈련된 모델을 'motion_classifier.h5' 파일로 저장했습니다.")

# Unity에서 사용하기 위해 ONNX 포맷으로 변환합니다.
model_proto, _ = tf2onnx.convert.from_keras(model)
with open("motion_classifier.onnx", "wb") as f:
    f.write(model_proto.SerializeToString())

print("모델을 Unity에서 사용할 'motion_classifier.onnx' 파일로 변환했습니다.")
print("\n🎉 모든 과정이 완료되었습니다! 이제 'motion_classifier.onnx' 파일을 Unity 프로젝트에 사용하세요.")