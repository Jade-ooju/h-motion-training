import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import json
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# --- 1. 데이터 로드 및 전처리 ---
print("🚀 앙상블 모델 생성기")
print("=" * 50)

data = np.load("preprocessed_data.npz")
X = data['X']
y = data['y']

print(f"📊 데이터 형태: {X.shape}")
print(f"📊 클래스 분포: {np.bincount(y)}")

# 시계열 데이터를 통계값으로 압축
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

# 특징 선택
print("\n🔍 특징 선택을 통한 차원 축소...")
selector = SelectKBest(score_func=f_classif, k=100)
X_selected = selector.fit_transform(X_compressed, y)
selected_features = selector.get_support(indices=True)
print(f"📊 선택된 특징 수: {len(selected_features)}")

# 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 훈련 데이터: {len(X_train)}개")
print(f"📊 검증 데이터: {len(X_val)}개")

# --- 2. 다양한 모델 훈련 ---
print("\n🌲 다양한 모델 훈련 시작...")

# Random Forest
print("🌲 Random Forest 훈련 중...")
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_val)
rf_accuracy = accuracy_score(y_val, rf_pred)

# Gradient Boosting
print("🌿 Gradient Boosting 훈련 중...")
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_val)
gb_accuracy = accuracy_score(y_val, gb_pred)

# SVM
print("🔧 SVM 훈련 중...")
svm_model = SVC(kernel='rbf', random_state=42, probability=True)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_val)
svm_accuracy = accuracy_score(y_val, svm_pred)

# --- 3. 앙상블 예측 ---
print("\n🎯 앙상블 예측 생성...")

# 각 모델의 예측 확률
rf_proba = rf_model.predict_proba(X_val)
gb_proba = gb_model.predict_proba(X_val)
svm_proba = svm_model.predict_proba(X_val)

# 가중 평균 앙상블 (Random Forest에 더 높은 가중치)
ensemble_proba = (0.5 * rf_proba + 0.3 * gb_proba + 0.2 * svm_proba)
ensemble_pred = np.argmax(ensemble_proba, axis=1)
ensemble_accuracy = accuracy_score(y_val, ensemble_pred)

# --- 4. 결과 비교 ---
print("\n📊 모델 성능 비교:")
print(f"🌲 Random Forest: {rf_accuracy:.4f}")
print(f"🌿 Gradient Boosting: {gb_accuracy:.4f}")
print(f"🔧 SVM: {gb_accuracy:.4f}")
print(f"🎯 앙상블: {ensemble_accuracy:.4f}")

# --- 5. 최고 성능 모델 선택 ---
models = {
    'Random Forest': (rf_model, rf_accuracy),
    'Gradient Boosting': (gb_model, gb_accuracy),
    'SVM': (svm_model, svm_accuracy),
    'Ensemble': (None, ensemble_accuracy)
}

best_model_name = max(models.keys(), key=lambda x: models[x][1])
best_accuracy = models[best_model_name][1]

print(f"\n🏆 최고 성능 모델: {best_model_name} (정확도: {best_accuracy:.4f})")

# --- 6. 최고 성능 모델 상세 분석 ---
if best_model_name == 'Ensemble':
    print("\n📋 앙상블 분류 보고서:")
    print(classification_report(y_val, ensemble_pred, target_names=['Pick', 'Hold', 'Place']))
else:
    best_model = models[best_model_name][0]
    best_pred = best_model.predict(X_val)
    print(f"\n📋 {best_model_name} 분류 보고서:")
    print(classification_report(y_val, best_pred, target_names=['Pick', 'Hold', 'Place']))

# --- 7. 특징 중요도 분석 ---
print("\n🔍 특징 중요도 분석...")

if best_model_name == 'Random Forest':
    feature_importance = rf_model.feature_importances_
elif best_model_name == 'Gradient Boosting':
    feature_importance = gb_model.feature_importances_
else:
    # SVM은 특징 중요도를 직접 제공하지 않으므로 Random Forest 사용
    feature_importance = rf_model.feature_importances_

top_features = np.argsort(feature_importance)[-10:]
print("🔍 상위 10개 중요 특징:")
for i, feature_idx in enumerate(reversed(top_features)):
    original_feature_idx = selected_features[feature_idx]
    importance = feature_importance[feature_idx]
    print(f"   {i+1:2d}. 선택된 특징 {feature_idx:3d} (원본 {original_feature_idx:3d}): {importance:.4f}")

# --- 8. 시각화 ---
plt.figure(figsize=(15, 5))

# 모델 성능 비교
plt.subplot(1, 3, 1)
model_names = list(models.keys())
accuracies = [models[name][1] for name in model_names]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
bars = plt.bar(model_names, accuracies, color=colors)
plt.title('Model Performance Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1.1)

# 정확도 값 표시
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{acc:.3f}', ha='center', va='bottom')

# 특징 중요도
plt.subplot(1, 3, 2)
plt.bar(range(len(feature_importance)), feature_importance)
plt.title(f'{best_model_name} Feature Importance')
plt.xlabel('Selected Feature Index')
plt.ylabel('Importance')

# 클래스별 정확도
plt.subplot(1, 3, 3)
if best_model_name == 'Ensemble':
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_val, ensemble_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Ensemble Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0, 1, 2], ['Pick', 'Hold', 'Place'])
    plt.yticks([0, 1, 2], ['Pick', 'Hold', 'Place'])
    
    # 숫자 표시
    for i in range(3):
        for j in range(3):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center')

plt.tight_layout()
plt.savefig('ensemble_model_results.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 9. 모델 저장 ---
print("\n💾 모델 저장 중...")

# 특징 선택 정보 저장
feature_selection_info = {
    "selected_features": selected_features.tolist(),
    "total_features": X_compressed.shape[1],
    "selected_count": len(selected_features),
    "compression_ratio": len(selected_features) / X_compressed.shape[1],
    "best_model": best_model_name,
    "best_accuracy": float(best_accuracy)
}

with open("ensemble_feature_selection_info.json", "w") as f:
    json.dump(feature_selection_info, f, indent=2)

print(f"✅ 특징 선택 정보를 'ensemble_feature_selection_info.json'에 저장했습니다.")

# 최고 성능 모델 저장
if best_model_name != 'Ensemble':
    import joblib
    joblib.dump(best_model, f"best_{best_model_name.lower().replace(' ', '_')}_model.pkl")
    print(f"✅ 최고 성능 모델을 'best_{best_model_name.lower().replace(' ', '_')}_model.pkl'로 저장했습니다.")

# --- 10. Unity 통합 정보 ---
print("\n🔧 Unity 통합 정보...")

unity_info = f"""
// Unity에서 사용할 모델 정보
// 최고 성능 모델: {best_model_name}
// 정확도: {best_accuracy:.4f}
// 입력 특징 수: {len(selected_features)}
// 
// 특징 선택 인덱스:
// {selected_features.tolist()}
// 
// 사용 방법:
// 1. Python에서 모델을 ONNX로 변환
// 2. Unity에서 ONNX Runtime 사용
// 3. 입력 데이터를 100개 특징으로 압축
// 4. 모델 예측 실행
"""

with open("unity_integration_info.txt", "w", encoding="utf-8") as f:
    f.write(unity_info)

print("✅ Unity 통합 정보를 'unity_integration_info.txt'에 저장했습니다.")

# --- 11. 요약 ---
print("\n" + "=" * 50)
print("🎉 앙상블 모델 생성 완료!")
print("=" * 50)
print(f"🏆 최고 성능 모델: {best_model_name}")
print(f"🎯 최고 정확도: {best_accuracy:.4f}")
print(f"📊 차원 감소율: {(1 - len(selected_features) / X_compressed.shape[1]) * 100:.1f}%")
print(f"💾 생성된 파일들:")
print(f"   - ensemble_feature_selection_info.json")
if best_model_name != 'Ensemble':
    print(f"   - best_{best_model_name.lower().replace(' ', '_')}_model.pkl")
print(f"   - unity_integration_info.txt")
print(f"   - ensemble_model_results.png")
print("\n🚀 이제 Unity에서 최고 성능 모델을 사용하세요!")


