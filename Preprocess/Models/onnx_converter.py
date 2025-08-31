#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX Model Converter for Unity Integration
"""

import joblib
import numpy as np
import json
import os
import onnxmltools
from onnxmltools.convert import convert_sklearn
from onnxmltools.utils import save_model

class ONNXConverter:
    def __init__(self, model_path="best_gradient_boosting_model.pkl",
                 feature_info_path="ensemble_feature_selection_info.json"):
        self.model_path = model_path
        self.feature_info_path = feature_info_path
        self.model = None
        self.feature_info = None
        
    def load_model_and_info(self):
        try:
            print(f"🔍 모델 로드 중: {self.model_path}")
            self.model = joblib.load(self.model_path)
            print(f"✅ 모델 로드 완료: {type(self.model).__name__}")
            
            print(f"🔍 특징 정보 로드 중: {self.feature_info_path}")
            with open(self.feature_info_path, 'r', encoding='utf-8') as f:
                self.feature_info = json.load(f)
            print(f"✅ 특징 정보 로드 완료")
            
        except Exception as e:
            print(f"❌ 로드 오류: {e}")
            raise
    
    def convert_to_onnx(self, output_path="best_gradient_boosting_model.onnx"):
        try:
            if self.model is None or self.feature_info is None:
                raise ValueError("모델과 특징 정보를 먼저 로드해야 합니다.")
            
            print(f"🔄 ONNX 변환 시작...")
            print(f"   모델 타입: {type(self.model).__name__}")
            
            # FloatTensorType 올바른 사용법
            from onnxmltools.convert.common.data_types import FloatTensorType
            input_shape = [1, len(self.feature_info['selected_features'])]
            initial_type = ('input', FloatTensorType(input_shape))
            
            print(f"   입력 형태: {input_shape}")
            
            # ONNX 변환
            onnx_model = convert_sklearn(self.model, initial_types=[initial_type])
            
            # 모델 저장
            save_model(onnx_model, output_path)
            print(f"✅ ONNX 변환 완료: {output_path}")
            
            # 정보 출력
            print(f"\n📋 ONNX 모델 정보:")
            print(f"   입력 형태: {input_shape}")
            print(f"   출력 클래스 수: 3")
            print(f"   파일 크기: {os.path.getsize(output_path) / 1024:.1f} KB")
            
            return output_path
            
        except Exception as e:
            print(f"❌ ONNX 변환 오류: {e}")
            raise
    
    def create_unity_code(self, onnx_path):
        unity_code = f"""// Unity C# ONNX Runtime Integration Code
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using UnityEngine;

public class MotionClassifier : MonoBehaviour
{{
    private InferenceSession session;
    private string modelPath = "{onnx_path}";
    
    void Start()
    {{
        try
        {{
            session = new InferenceSession(modelPath);
            Debug.Log("ONNX 모델 로드 성공");
        }}
        catch (Exception e)
        {{
            Debug.LogError($"ONNX 모델 로드 실패: {{e.Message}}");
        }}
    }}
    
    public string PredictMotion(float[] features)
    {{
        try
        {{
            var inputName = session.InputNames[0];
            var inputTensor = new DenseTensor<float>(features, new int[] {{ 1, {len(self.feature_info['selected_features'])} }});
            
            var inputs = new List<NamedOnnxValue>
            {{
                NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
            }};
            
            using (var results = session.Run(inputs))
            {{
                var output = results.First();
                var outputTensor = output.AsTensor<float>();
                
                var probabilities = new float[outputTensor.Dimensions[1]];
                for (int i = 0; i < probabilities.Length; i++)
                {{
                    probabilities[i] = outputTensor[0, i];
                }}
                
                int predictedClass = 0;
                float maxProb = probabilities[0];
                for (int i = 1; i < probabilities.Length; i++)
                {{
                    if (probabilities[i] > maxProb)
                    {{
                        maxProb = probabilities[i];
                        predictedClass = i;
                    }}
                }}
                
                string[] classNames = {{ "Pick", "Hold", "Place" }};
                return classNames[predictedClass];
            }}
        }}
        catch (Exception e)
        {{
            Debug.LogError($"예측 실패: {{e.Message}}");
            return "Unknown";
        }}
    }}
    
    void OnDestroy()
    {{
        session?.Dispose();
    }}
}}"""
        
        unity_code_path = "unity_onnx_integration.cs"
        with open(unity_code_path, 'w', encoding='utf-8') as f:
            f.write(unity_code)
        
        print(f"✅ Unity 통합 코드 생성: {unity_code_path}")
        return unity_code_path

def main():
    print("🚀 ONNX Model Converter for Unity Integration")
    print("=" * 60)
    
    try:
        converter = ONNXConverter()
        converter.load_model_and_info()
        
        onnx_path = converter.convert_to_onnx()
        unity_code_path = converter.create_unity_code(onnx_path)
        
        print(f"\n🎉 ONNX 변환 완료!")
        print(f"📁 생성된 파일들:")
        print(f"   - {onnx_path}")
        print(f"   - {unity_code_path}")
        print(f"\n🚀 Unity에서 ONNX Runtime을 사용할 수 있습니다!")
        
    except Exception as e:
        print(f"❌ 변환 실패: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()

