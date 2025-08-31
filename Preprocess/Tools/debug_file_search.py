#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug File Search
H2O 파일 검색이 제대로 작동하는지 디버깅하는 스크립트
"""

import os
import glob

def debug_file_search():
    """H2O 파일 검색을 디버깅합니다."""
    print("🔍 H2O 파일 검색 디버깅...")
    print("=" * 50)
    
    # 현재 작업 디렉토리
    current_dir = os.getcwd()
    print(f"📁 현재 작업 디렉토리: {current_dir}")
    
    # Data 폴더 경로 (프로젝트 루트)
    data_dir = os.path.join(current_dir, "..", "..", "Data")
    data_dir_abs = os.path.abspath(data_dir)
    print(f"📁 Data 폴더 경로: {data_dir}")
    print(f"📁 Data 폴더 절대 경로: {data_dir_abs}")
    
    # Data 폴더 존재 확인
    if not os.path.exists(data_dir_abs):
        print(f"❌ Data 폴더가 존재하지 않습니다!")
        return
    
    # 파일 검색 테스트
    print(f"\n🔍 파일 검색 테스트:")
    
    # 1. 모든 JSON 파일
    all_json_pattern = os.path.join(data_dir_abs, "*.json")
    all_json_files = glob.glob(all_json_pattern)
    print(f"   📄 모든 JSON 파일 (*.json): {len(all_json_files)}개")
    print(f"   🔍 검색 패턴: {all_json_pattern}")
    
    # 2. H2O JSON 파일
    h2o_pattern = os.path.join(data_dir_abs, "*_h2o_*.json")
    h2o_files = glob.glob(h2o_pattern)
    print(f"   📄 H2O JSON 파일 (*_h2o_*.json): {len(h2o_files)}개")
    print(f"   🔍 검색 패턴: {h2o_pattern}")
    
    # 3. 기존 JSON 파일 (H2O 제외)
    existing_files = []
    for file_path in all_json_files:
        filename = os.path.basename(file_path)
        if not filename.startswith(("Hold_h2o_", "Pick_h2o_", "Place_h2o_")):
            existing_files.append(file_path)
    
    print(f"   📄 기존 JSON 파일 (H2O 제외): {len(existing_files)}개")
    
    # 4. 파일 목록 출력
    print(f"\n📋 H2O 파일 목록 (처음 10개):")
    for i, file_path in enumerate(h2o_files[:10]):
        print(f"   {i+1:2d}. {os.path.basename(file_path)}")
    
    if len(h2o_files) > 10:
        print(f"   ... 및 {len(h2o_files) - 10}개 더")
    
    # 5. 총 파일 수
    total_files = len(existing_files) + len(h2o_files)
    print(f"\n📊 총 파일 수: {total_files}개")
    print(f"   - 기존 파일: {len(existing_files)}개")
    print(f"   - H2O 파일: {len(h2o_files)}개")
    
    print(f"\n✅ 디버깅 완료!")

if __name__ == "__main__":
    debug_file_search()
