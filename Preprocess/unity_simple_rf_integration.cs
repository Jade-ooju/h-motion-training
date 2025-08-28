
// Unity C# 간단한 Random Forest 대체 모델 통합 코드
// 이 코드를 Unity 스크립트에 추가하세요

using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class SimpleRFMotionClassifier
{
    // 선택된 특징 인덱스 (Python에서 생성된 정보)
    private static readonly int[] SelectedFeatures = {
        1, 4, 7, 29, 57, 60, 63, 85, 88, 91, 93, 113, 116, 119, 121, 141, 144, 147, 169, 172, 175, 197, 200, 203, 205, 209, 225, 228, 231, 233, 239, 253, 256, 259, 261, 267, 281, 284, 287, 289, 295, 309, 315, 337, 340, 343, 345, 365, 368, 371, 373, 379, 393, 396, 399, 401, 407, 410, 421, 424, 427, 429, 435, 438, 449, 455, 477, 480, 483, 505, 508, 511, 519, 522, 533, 536, 539, 547, 550, 561, 564, 567, 575, 578, 589, 595, 617, 620, 623, 645, 648, 651, 673, 676, 679, 687, 701, 704, 707, 715
    };
    
    // 시계열 데이터를 통계값으로 압축
    public static float[] CompressSequence(List<Vector3[]> jointPositions, List<Quaternion[]> jointRotations)
    {
        int totalFeatures = jointPositions[0].Length * 7; // position(3) + rotation(4)
        float[] compressedFeatures = new float[totalFeatures * 4]; // 각 특징별로 4개의 통계값
        
        // 각 특징별로 통계값 계산
        for (int featureIdx = 0; featureIdx < totalFeatures; featureIdx++)
        {
            List<float> values = new List<float>();
            
            // 각 프레임에서 해당 특징 값 추출
            for (int frameIdx = 0; frameIdx < jointPositions.Count; frameIdx++)
            {
                if (featureIdx < jointPositions[frameIdx].Length * 3)
                {
                    // Position 데이터
                    int jointIdx = featureIdx / 3;
                    int componentIdx = featureIdx % 3;
                    if (jointIdx < jointPositions[frameIdx].Length)
                        values.Add(jointPositions[frameIdx][jointIdx][componentIdx]);
                }
                else
                {
                    // Rotation 데이터
                    int jointIdx = (featureIdx - jointPositions[frameIdx].Length * 3) / 4;
                    int componentIdx = (featureIdx - jointPositions[frameIdx].Length * 3) % 4;
                    if (jointIdx < jointRotations[frameIdx].Length)
                        values.Add(jointRotations[frameIdx][jointIdx][componentIdx]);
                }
            }
            
            // 통계값 계산 (평균, 표준편차, 최대, 최소)
            if (values.Count > 0)
            {
                compressedFeatures[featureIdx * 4] = values.Average();           // 평균
                compressedFeatures[featureIdx * 4 + 1] = CalculateStd(values);  // 표준편차
                compressedFeatures[featureIdx * 4 + 2] = values.Max();          // 최대
                compressedFeatures[featureIdx * 4 + 3] = values.Min();          // 최소
            }
        }
        
        // 선택된 특징만 반환
        float[] selectedFeatures = new float[SelectedFeatures.Length];
        for (int i = 0; i < SelectedFeatures.Length; i++)
        {
            selectedFeatures[i] = compressedFeatures[SelectedFeatures[i]];
        }
        
        return selectedFeatures;
    }
    
    // 표준편차 계산
    private static float CalculateStd(List<float> values)
    {
        if (values.Count == 0) return 0f;
        float mean = values.Average();
        float sumSquaredDiff = values.Sum(x => (x - mean) * (x - mean));
        return Mathf.Sqrt(sumSquaredDiff / values.Count);
    }
    
    // 동작 분류 (ONNX 모델 사용)
    public static string ClassifyMotion(float[] inputFeatures)
    {
        // TODO: ONNX Runtime for Unity를 사용하여 모델 예측
        // float[] predictions = onnxModel.Predict(inputFeatures);
        // int predictedClass = System.Array.IndexOf(predictions, predictions.Max());
        
        string[] classNames = { "Pick", "Hold", "Place" };
        // return classNames[predictedClass];
        
        return "Hold"; // 임시 반환값
    }
    
    // 모델 정보
    public static string GetModelInfo()
    {
        return $"간단한 Random Forest 대체 모델\n" +
               $"입력 특징 수: {SelectedFeatures.Length}\n" +
               $"원본 Random Forest 정확도: 100.0%\n" +
               $"대체 모델 정확도: 100.0%";
    }
}

// 사용 예시:
// List<Vector3[]> allJointPositions = ...; // 모든 프레임의 관절 위치 데이터
// List<Quaternion[]> allJointRotations = ...; // 모든 프레임의 관절 회전 데이터
// float[] inputFeatures = SimpleRFMotionClassifier.CompressSequence(allJointPositions, allJointRotations);
// string predictedMotion = SimpleRFMotionClassifier.ClassifyMotion(inputFeatures);
