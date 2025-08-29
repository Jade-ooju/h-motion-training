#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check Preprocessed Data
전처리된 데이터의 정보를 확인하는 스크립트
"""

import numpy as np
import os

def check_preprocessed_data():
    """전처리된 데이터의 정보를 확인합니다."""
    print("🔍 전처리된 데이터 정보 확인...")
    print("=" * 50)
    
    # 전처리된 데이터 파일 경로
    data_file = "../../preprocessed_data.npz"
    
    if not os.path.exists(data_file):
        print(f"❌ 전처리된 데이터 파일을 찾을 수 없습니다: {data_file}")
        return False
    
    try:
        # 데이터 로드
        data = np.load(data_file)
        
        print(f"📊 전처리된 데이터 정보:")
        print(f"   - 파일 크기: {os.path.getsize(data_file) / (1024 * 1024):.2f} MB")
        print(f"   - 저장된 키들: {list(data.keys())}")
        
        # 각 키의 정보 확인
        for key in data.keys():
            if hasattr(data[key], 'shape'):
                print(f"   - {key}: {data[key].shape}")
            else:
                print(f"   - {key}: {type(data[key])}")
        
        # 클래스 분포 확인 (가능한 경우)
        if 'y_train' in data:
            unique, counts = np.unique(data['y_train'], return_counts=True)
            print(f"\n🏷️ 훈련 데이터 클래스 분포: {dict(zip(unique, counts))}")
        
        if 'y_val' in data:
            unique, counts = np.unique(data['y_val'], return_counts=True)
            print(f"🏷️ 검증 데이터 클래스 분포: {dict(zip(unique, counts))}")
        
        print(f"\n✅ 데이터 확인 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 데이터 확인 오류: {e}")
        return False

if __name__ == "__main__":
    check_preprocessed_data()
