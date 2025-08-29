#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H2O Data Quality Checker
변환된 H2O 데이터의 품질을 검증하는 스크립트
"""

import json
import os
import glob
from collections import Counter

def check_h2o_data_quality():
    """H2O 데이터 품질을 검증합니다."""
    print("🔍 H2O 데이터 품질 검증 시작...")
    print("=" * 50)
    
    # H2O JSON 파일 찾기
    json_files = glob.glob("../../Data/*_h2o_*.json")
    print(f"📊 총 H2O JSON 파일 수: {len(json_files)}")
    
    if not json_files:
        print("❌ H2O JSON 파일을 찾을 수 없습니다!")
        return False
    
    # 라벨별 분포 확인
    labels = [os.path.basename(f).split('_')[0] for f in json_files]
    label_counts = Counter(labels)
    print(f"🏷️ 라벨별 분포: {dict(label_counts)}")
    
    # 샘플 파일 구조 분석
    sample_file = json_files[0]
    print(f"\n📋 샘플 파일 분석: {os.path.basename(sample_file)}")
    
    try:
        with open(sample_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"   ✅ JSON 파싱 성공")
        print(f"   📝 sessionId: {data.get('sessionId', 'N/A')}")
        print(f"   🎯 interactionTargetObject: {data.get('interactionTargetObject', 'N/A')}")
        print(f"   📍 interactionTargetPosition: {data.get('interactionTargetPosition', 'N/A')}")
        print(f"   🎬 frames 수: {len(data.get('frames', []))}")
        
        if data.get('frames'):
            first_frame = data['frames'][0]
            joints = first_frame.get('joints', [])
            print(f"   🦴 첫 번째 프레임 관절 수: {len(joints)}")
            
            if joints:
                first_joint = joints[0]
                print(f"   🔍 첫 번째 관절 구조:")
                print(f"      - jointName: {first_joint.get('jointName', 'N/A')}")
                print(f"      - position: {first_joint.get('position', 'N/A')}")
                print(f"      - rotation: {first_joint.get('rotation', 'N/A')}")
                print(f"      - confidence: {first_joint.get('confidence', 'N/A')}")
        
        # 파일 크기 통계
        sizes = [os.path.getsize(f) for f in json_files]
        print(f"\n📏 파일 크기 통계:")
        print(f"   - 최소: {min(sizes):,} bytes")
        print(f"   - 최대: {max(sizes):,} bytes")
        print(f"   - 평균: {sum(sizes)//len(sizes):,} bytes")
        
        print(f"\n✅ 데이터 품질 검증 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 데이터 검증 오류: {e}")
        return False

if __name__ == "__main__":
    check_h2o_data_quality()
