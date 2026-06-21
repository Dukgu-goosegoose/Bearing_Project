"""
CWRU 데이터 정리 스크립트 (로컬 / Windows PowerShell 용)

하는 일:
  1. XiongMeijing/CWRU-1 레포를 받는다 (파일명이 이미 IR007_0 같은 라벨 이름).
  2. 네 프로젝트의 data/raw/ 폴더 구조에 맞게 '평면 복사'한다.
     (파일명은 그대로 유지 → 네 data_loader.py 의 정규식에 바로 걸림)
  3. 임시 폴더는 지운다.

실행 방법:
  - 네 프로젝트 '루트'(data/ 폴더가 보이는 위치)에서 실행한다.
  - PowerShell:
        python organize_cwru.py
  - git 이 설치돼 있어야 한다 (git --version 으로 확인).
"""

import shutil
import subprocess
from pathlib import Path

# ============================================================
# 설정 — 필요하면 여기만 고친다
# ============================================================
REPO_URL  = "https://github.com/XiongMeijing/CWRU-1.git"
CLONE_DIR = Path("_cwru_tmp")        # 임시로 레포를 받을 위치
RAW_DIR   = Path("data") / "raw"     # 네 프로젝트의 data/raw

# 레포 폴더  ->  네 프로젝트 폴더 (config.yaml 의 경로와 일치시킴)
#   Normal  : 정상(비지도 학습용)
#   12k_DE  : 메인 고장 데이터 (Drive End 12kHz)
#   48k_DE  : 고해상도 고장 데이터 (Drive End 48kHz)
#   12k_FE  : Fan End — 베어링이 6203이라 물리진단은 별도 처리 필요(주의)
MAPPING = {
    "Normal": RAW_DIR / "normal",
    "12k_DE": RAW_DIR / "fault_12k",
    "48k_DE": RAW_DIR / "fault_48k",
    "12k_FE": RAW_DIR / "fan_end_fault",
}
# ============================================================


def main() -> None:
    # 1) 레포 받기 (이미 있으면 다시 안 받음)
    if CLONE_DIR.exists():
        print(f"[건너뜀] {CLONE_DIR} 가 이미 있음 — 기존 것 사용")
    else:
        print(f"[clone] {REPO_URL} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(CLONE_DIR)],
            check=True,
        )

    src_data = CLONE_DIR / "Data"
    if not src_data.exists():
        raise SystemExit(f"[오류] {src_data} 가 없음. 레포 구조가 바뀌었는지 확인.")

    # 2) 라벨 이름 그대로 평면 복사
    print("\n=== 복사 결과 ===")
    total = 0
    for sub, dst in MAPPING.items():
        src = src_data / sub
        if not src.exists():
            print(f"[건너뜀] {src} 없음")
            continue
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for mat in sorted(src.glob("*.mat")):
            shutil.copy(mat, dst / mat.name)   # 파일명 그대로 유지
            n += 1
        total += n
        print(f"  {sub:8s} -> {dst}   ({n}개)")
    print(f"총 {total}개 복사 완료")

    # 3) 임시 폴더 정리
    shutil.rmtree(CLONE_DIR, ignore_errors=True)
    print(f"[정리] {CLONE_DIR} 삭제\n")

    # 4) 라벨이 잘 읽히는지 미리보기 (네 코드 정규식과 동일한 확인)
    import re
    rx = re.compile(r"(IR|OR|B)(\d{3})", re.IGNORECASE)
    print("=== 라벨 인식 미리보기 (fault_12k 앞 8개) ===")
    sample_dir = RAW_DIR / "fault_12k"
    for mat in sorted(sample_dir.glob("*.mat"))[:8]:
        m = rx.search(mat.stem.upper())
        loc = m.group(1) if m else "UNKNOWN(!)"
        load = mat.stem.split("_")[-1]
        print(f"  {mat.name:16s} -> 위치={loc}, 부하={load}")


if __name__ == "__main__":
    main()
