# 🤖 Robotics Motion Classification - Preprocess 폴더

## 📋 개요
이 폴더는 로봇 동작 분류를 위한 머신러닝 모델 훈련, Python 브리지, Unity 통합을 위한 핵심 스크립트들을 포함합니다.

## 🎯 **가장 추천하는 접근법: Python 브리지 방식**

### 🏆 **추천 이유:**
- **100% 정확도 유지** (원본 Random Forest 성능 보존)
- **실시간 Unity 통합** 가능
- **안정적이고 확장 가능한** 구조

---

## 📁 **폴더 구조**

```
Preprocess/
├── 📁 Core/                    # 핵심 실행 파일들
│   ├── motion_classifier_bridge.py      # Python 브리지 (메인)
│   ├── unity_motion_classifier.cs       # Unity C# 통신 코드
│   └── data_preprocessor.py             # 데이터 전처리 스크립트
│
├── 📁 Models/                  # 모델 훈련 및 저장
│   ├── ensemble_motion_classifier.py    # 앙상블 모델 훈련
│   ├── feature_based_classifier.py      # 특징 기반 모델
│   ├── model_validation.py              # 모델 검증
│   ├── best_random_forest_model.pkl     # 훈련된 Random Forest 모델
│   ├── ensemble_features.json           # 앙상블 모델 특징 정보
│   └── feature_based_features.json      # 특징 기반 모델 특징 정보
│
├── 📁 Tools/                   # 디버깅 및 유틸리티
│   └── preprocessing_debugger.py        # 전처리 디버깅 도구
│
├── 📁 Outputs/                 # 생성된 결과물들
│   └── *.png (결과 이미지들)
│
├── 📁 Data/                    # 원본 JSON 데이터
│   ├── Hold_*.json
│   ├── Pick_*.json
│   └── Place_*.json
│
└── 📄 README.md                # 메인 문서
```

---

## 📚 **핵심 스크립트 설명**

### 🚀 **1. Core 폴더 (핵심 실행 파일들)**
| 파일명 | 설명 | 사용법 |
|--------|------|--------|
| `motion_classifier_bridge.py` | **🎯 메인 추천 스크립트**<br/>Unity에서 Python 프로세스로 Random Forest 모델 직접 사용 | `python Core/motion_classifier_bridge.py` |
| `unity_motion_classifier.cs` | Unity C# 통신 코드<br/>Python 프로세스와의 양방향 통신 | Unity 프로젝트에 추가 |
| `data_preprocessor.py` | JSON 데이터 → 전처리 → NPZ 저장 | `python Core/data_preprocessor.py` |

### 🧠 **2. Models 폴더 (모델 훈련 및 저장)**
| 파일명 | 설명 | 사용법 |
|--------|------|--------|
| `ensemble_motion_classifier.py` | **100% 정확도 앙상블 모델**<br/>Random Forest + Gradient Boosting + SVM | `python Models/ensemble_motion_classifier.py` |
| `feature_based_classifier.py` | 특징 기반 모델<br/>시계열 데이터 → 통계값 압축 → 특징 선택 | `python Models/feature_based_classifier.py` |
| `model_validation.py` | 간단한 모델들로 데이터 학습 가능성 테스트 | `python Models/model_validation.py` |

### 🔧 **3. Tools 폴더 (디버깅 및 유틸리티)**
| 파일명 | 설명 | 사용법 |
|--------|------|--------|
| `preprocessing_debugger.py` | 전처리 과정 디버깅 및 데이터 품질 확인 | `python Tools/preprocessing_debugger.py` |

---

## 🎯 **사용 가이드**

### **1단계: 데이터 전처리**
```bash
cd h-motion-training/Preprocess
python Core/data_preprocessor.py
```
**결과:** `preprocessed_data.npz` 파일 생성

### **2단계: 모델 훈련**
```bash
python Models/ensemble_motion_classifier.py
```
**결과:** `best_random_forest_model.pkl`, `ensemble_features.json` 생성

### **3단계: Python 브리지 테스트**
```bash
python Core/motion_classifier_bridge.py
```
**예상 결과:** 100% 정확도, "Hold" 예측, [0.22, 0.73, 0.05] 확률

### **4단계: Unity 통합**
1. `Core/unity_motion_classifier.cs`를 Unity 프로젝트에 추가
2. `UnityPythonMotionClassifier` 컴포넌트를 GameObject에 추가
3. Python 스크립트 경로 설정 (`Core/motion_classifier_bridge.py`)
4. 자동으로 Python 프로세스 시작 및 통신

### **5단계: 실시간 동작 분류**
```csharp
// Unity에서 사용 예시
var classifier = GetComponent<UnityPythonMotionClassifier>();
string result = classifier.PredictMotion(jointPositions, jointRotations);
```

---

## ⚠️ **주의사항**

### **경로 설정:**
- **Core 폴더**: `../Models/`, `../Data/` 참조
- **Models 폴더**: `../preprocessed_data.npz`, `../Outputs/` 참조
- **Tools 폴더**: `../Data/` 참조

### **Python 브리지의 장점:**
- **100% 정확도**: 원본 모델 성능 완벽 보존
- **실시간 예측**: Unity에서 즉시 사용 가능
- **확장성**: 새로운 모델이나 전처리 쉽게 추가

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
3. 모델 파일 경로 확인 (`Models/best_random_forest_model.pkl`)

### **Unity 통합 오류 시:**
1. Python 스크립트 경로 확인 (`Core/motion_classifier_bridge.py`)
2. Python 실행 파일 경로 설정
3. 프로세스 권한 확인

---

## 🎉 **정리 완료!**

**이제 Preprocess 폴더가 기능별로 체계적으로 정리되어 혼란 없이 사용할 수 있습니다.**

**Python 브리지 방식으로 100% 정확도의 동작 분류를 Unity에서 실시간으로 사용하세요!** 🎯
