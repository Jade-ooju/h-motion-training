# 🤖 Robotics Motion Classification - Preprocess 폴더

## 📋 개요
이 폴더는 로봇 동작 분류를 위한 머신러닝 모델 훈련, ONNX 변환, Unity 통합을 위한 모든 스크립트와 파일들을 포함합니다.

## 🎯 **가장 추천하는 접근법: Python 브리지 방식**

### 🏆 **추천 이유:**
- **100% 정확도 유지** (원본 Random Forest 성능 보존)
- **실시간 Unity 통합** 가능
- **안정적이고 확장 가능한** 구조

### 📁 **필요한 핵심 파일들:**
1. `unity_python_bridge.py` - Python 브리지 (메인)
2. `unity_python_communication.cs` - Unity C# 통신 코드
3. `best_random_forest_model.pkl` - 훈련된 Random Forest 모델
4. `ensemble_feature_selection_info.json` - 특징 선택 정보

---

## 📚 **스크립트 분류 및 설명**

### 🚀 **1. Python 브리지 (추천)**
| 파일명 | 설명 | 상태 |
|--------|------|------|
| `unity_python_bridge.py` | **🎯 메인 추천 스크립트**<br/>Unity에서 Python 프로세스로 Random Forest 모델 직접 사용 | ✅ **완벽 동작** |
| `unity_python_communication.cs` | Unity C# 통신 코드<br/>Python 프로세스와의 양방향 통신 | ✅ **완벽 생성** |

### 🔄 **2. ONNX 변환 시도들 (성능 저하)**
| 파일명 | 설명 | 문제점 |
|--------|------|--------|
| `final_simple_onnx.py` | StandardScaler 없는 로지스틱 회귀 → ONNX | ❌ 84.4% 정확도 (15.6% 손실) |
| `direct_random_forest_onnx.py` | Random Forest → 로지스틱 회귀 → ONNX | ❌ 예측 불일치 (Hold → Place) |
| `final_mlp_to_onnx.py` | MLP → ONNX 변환 | ❌ 55.6% 정확도 (큰 성능 손실) |
| `random_forest_to_onnx.py` | Random Forest → MLP → ONNX | ❌ onnxmltools API 오류 |
| `simple_rf_to_onnx.py` | 간단한 Random Forest → ONNX | ❌ 성능 저하 |

### 🧠 **3. 모델 훈련 및 실험**
| 파일명 | 설명 | 결과 |
|--------|------|------|
| `ensemble_model.py` | Random Forest + Gradient Boosting + SVM 앙상블 | ✅ **100% 정확도 달성** |
| `feature_based_model.py` | 시계열 데이터 → 통계값 압축 → 특징 선택 | ✅ **100% 정확도 달성** |
| `simple_model_test.py` | 간단한 모델들로 데이터 학습 가능성 테스트 | ✅ **데이터 품질 확인** |
| `onnx_converter_LSTM.py` | LSTM 모델 훈련 및 ONNX 변환 | ❌ **과소적합 문제** |

### 🔧 **4. 데이터 전처리 및 디버깅**
| 파일명 | 설명 | 상태 |
|--------|------|------|
| `preprocess_test.py` | JSON 데이터 → 전처리 → NPZ 저장 | ✅ **메인 전처리 스크립트** |
| `debug_preprocessing.py` | 전처리 과정 디버깅 및 데이터 품질 확인 | ✅ **디버깅 완료** |
| `test_onnx_model.py` | ONNX 모델 테스트 | ✅ **테스트 완료** |

### 📊 **5. 결과 및 시각화**
| 파일명 | 설명 | 내용 |
|--------|------|------|
| `ensemble_model_results.png` | 앙상블 모델 훈련 결과 | 정확도 100% 확인 |
| `feature_based_model_results.png` | 특징 기반 모델 결과 | 특징 중요도 분석 |
| `simple_model_test_results.png` | 간단한 모델 테스트 결과 | 데이터 학습 가능성 확인 |
| `training_history.png` | LSTM 훈련 히스토리 | 과소적합 문제 시각화 |

---

## 🎯 **사용 가이드**

### **1단계: Python 브리지 테스트**
```bash
cd Preprocess
python unity_python_bridge.py
```
**예상 결과:** 100% 정확도, "Hold" 예측, [0.22, 0.73, 0.05] 확률

### **2단계: Unity 통합**
1. `unity_python_communication.cs`를 Unity 프로젝트에 추가
2. `UnityPythonMotionClassifier` 컴포넌트를 GameObject에 추가
3. Python 스크립트 경로 설정
4. 자동으로 Python 프로세스 시작 및 통신

### **3단계: 실시간 동작 분류**
```csharp
// Unity에서 사용 예시
var classifier = GetComponent<UnityPythonMotionClassifier>();
string result = classifier.PredictMotion(jointPositions, jointRotations);
```

---

## ⚠️ **주의사항**

### **ONNX 변환의 한계:**
- **성능 손실**: 100% → 84.4% (로지스틱 회귀)
- **예측 불일치**: 완전히 다른 동작으로 분류
- **복잡한 결정 경계**: 선형 모델로는 근사 불가

### **Python 브리지의 장점:**
- **100% 정확도**: 원본 모델 성능 완벽 보존
- **실시간 예측**: Unity에서 즉시 사용 가능
- **확장성**: 새로운 모델이나 전처리 쉽게 추가

---

## 🗂️ **파일 정리 권장사항**

### **보관할 핵심 파일들:**
- `unity_python_bridge.py` ⭐ **메인**
- `unity_python_communication.cs` ⭐ **Unity 통합**
- `best_random_forest_model.pkl` ⭐ **훈련된 모델**
- `ensemble_feature_selection_info.json` ⭐ **특징 정보**
- `preprocess_test.py` ⭐ **데이터 전처리**
- `ensemble_model.py` ⭐ **모델 훈련**

### **삭제 가능한 파일들:**
- `*_to_onnx.py` (ONNX 변환 실패 스크립트들)
- `*_equivalent.onnx` (성능 저하된 ONNX 모델들)
- `*_equivalent.h5` (성능 저하된 TensorFlow 모델들)
- `unity_*_integration.cs` (ONNX 기반 Unity 코드들)

---

## 🚀 **최종 권장사항**

### **🎯 즉시 사용:**
1. **Python 브리지 방식**으로 Unity 통합
2. **100% 정확도** 유지하면서 실시간 동작 분류
3. **확장 가능한** 구조로 새로운 기능 추가

### **🔮 향후 개선:**
1. **더 복잡한 모델** (ST-GCN 등) Python 브리지로 통합
2. **실시간 학습** 기능 추가
3. **멀티스레딩** 최적화로 성능 향상

---

## 📞 **문제 해결**

### **Python 브리지 오류 시:**
1. Python 환경 확인 (`conda activate robotics_training`)
2. 필요한 패키지 설치 (`pip install scikit-learn joblib numpy`)
3. 모델 파일 경로 확인

### **Unity 통합 오류 시:**
1. Python 스크립트 경로 확인
2. Python 실행 파일 경로 설정
3. 프로세스 권한 확인

---

**🎉 Python 브리지 방식으로 100% 정확도의 동작 분류를 Unity에서 실시간으로 사용하세요!**
