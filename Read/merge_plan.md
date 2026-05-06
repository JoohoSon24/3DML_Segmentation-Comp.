# GitHub → 로컬 머지 계획

## 목적

GitHub 메인 브랜치(JoohoSon24/3DML_Segmentation-Comp)의 개선된 배치 로직을 베이스로,
로컬에서 작업한 surface sampling, SoftGroup 환경, 분석 문서들을 통합한다.

---

## 현황 파악

### GitHub main 구조 (확인 완료)

SoftGroup이 `third_party/`가 아니라 **레포 루트의 `softgroup/` 폴더**로 직접 통합되어 있음.

```
softgroup/
  data/nubzuki.py          ← Nubzuki 데이터셋 클래스 (이미 구현됨)
  model/softgroup.py       ← SoftGroup 모델
  ops/ops.*.so             ← cu124로 컴파일된 바이너리 (cu126에서 재빌드 필요)
  ops/src/                 ← CUDA 소스 포함
  util/, evaluation/       ← 유틸/평가 코드
configs/softgroup/
  softgroup_nubzuki.yaml   ← Nubzuki용 config (이미 있음)
  nubzuki_multiscan_*.yaml ← config 2개 더
tools/
  train.py, test.py        ← 학습/추론 스크립트
dataset_tools/
  npy_to_pth.py            ← .npy → SoftGroup .pth 변환 (이미 구현됨)
setup.py                   ← 레포 루트에서 ops 빌드
train_nubzuki_softgroup_x2.sh  ← 학습 shell 스크립트
README_SoftGroup_cu124_install.md  ← cu124 기준 설치 가이드
```

**중요**: 레포에 `ops.cpython-310-x86_64-linux-gnu.so`가 cu124로 컴파일된 채 커밋되어 있음.
cu126 환경에서는 반드시 `setup.py`로 재빌드해야 함.

### GitHub main이 로컬보다 앞서 있는 것
- `generate_synthetic_dataset.py` 전체 리팩토링:
  - HSV 컬러 지터링 (hue/saturation/value 독립 조절)
  - Layout mode 시스템 (scene_only 35% / mixed 45% / stack_heavy 20%)
  - Stacking style (centered / edge_overhang / corner_overhang)
  - AABB 충돌 검사에 epsilon tolerance + support parent 제외
  - 씬 생성 실패 시 최대 5회 재시도
- SoftGroup 전체 통합 (`softgroup/`, `configs/`, `tools/`, `setup.py`)
- `dataset_tools/npy_to_pth.py` 변환 스크립트
- 학습/추론 shell 스크립트들

### 로컬(instantmesh 브랜치)이 GitHub보다 앞서 있는 것
- `dataset_tools/generate_synthetic_dataset_v2.py` (surface sampling, OBJECT_SAMPLE_COUNT_RANGE)
- `Read/softgroup_integration_plan.md` (최신 버전)
- `Read/sampling_gap_analysis.md`
- `SoftGroup_cu126_install.md` (cu126 기준, GitHub의 cu124 가이드보다 최신)

---

## 단계별 계획

### Phase 1: 새 디렉토리에 GitHub 레포 클론

```bash
cd /home/ubuntu/cs479
git clone https://github.com/JoohoSon24/3DML_Segmentation-Comp.git 3DML_Merged
cd 3DML_Merged
```

### Phase 2: 새 conda 환경 생성

`SoftGroup_cu126_install.md` 전체 순서대로 새 환경을 처음부터 세팅한다.
환경 이름: `3d-seg` (README 기본값 그대로)

```bash
# 1) 환경 생성
conda create -n 3d-seg python=3.10 -y
conda activate 3d-seg

# 2) PyTorch + 기본 패키지
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu126
pip install numpy scipy tqdm matplotlib
conda install "mkl<2024.1" -y

# 3) CUDA 12.4 toolkit (custom ops 빌드용)
conda install -y cuda-toolkit=12.4 -c nvidia
export CUDA_HOME="$CONDA_PREFIX"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"

# 4) spconv
pip install spconv-cu126

# 5) SoftGroup 추가 의존성
pip install munch pyyaml tensorboard tensorboardX scikit-learn plyfile pandas six trimesh

# 6) sparsehash (SoftGroup ops 빌드용)
conda install -y -c conda-forge sparsehash

# 7) SoftGroup ops 빌드 (3DML_Merged 기준)
cd /home/ubuntu/cs479/3DML_Merged/third_party/SoftGroup
pip install -e . --no-build-isolation

# 8) 동작 확인
python - <<'PY'
import torch, spconv.pytorch as spconv
from softgroup.model import SoftGroup
print('ok', torch.__version__, torch.cuda.is_available())
PY
```

### Phase 3: SoftGroup ops 재빌드 (cu126용)

GitHub 레포에 cu124로 컴파일된 `.so` 파일이 포함되어 있어 그대로 사용 불가.
클론 후 레포 루트의 `setup.py`로 재빌드.

```bash
cd /home/ubuntu/cs479/3DML_Merged
export CUDA_HOME="$CONDA_PREFIX"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"
pip install -e . --no-build-isolation

# 확인
python -c "from softgroup.ops import ops; print('ops ok')"
```

### Phase 4: `generate_synthetic_dataset.py` 머지 (핵심)

GitHub 버전을 베이스로 우리의 surface sampling을 이식한다.

**변경 목표:**
- GitHub의 placement 로직(layout mode, stacking style, HSV 컬러) 그대로 유지
- `_sample_object_points` 함수만 우리 v2 버전으로 교체:
  - `mesh.vertices` + voxel downsampling 제거
  - `trimesh.sample.sample_surface(mesh, count)` + barycentric normal/color 보간으로 대체
  - `OBJECT_SAMPLE_COUNT_RANGE = (3000, 26000)` 추가
- `PlacedObject` dataclass: `target_point_spacing`, `spacing_ratio`, `raw_vertex_count` 제거 → `sample_count` 추가
- manifest: 동일하게 `sample_count`로 업데이트

**확인해야 할 의존 관계:**
- GitHub 버전 `_place_object`가 `_sample_object_points`를 어떻게 호출하는지 확인
- `scene_point_spacing` 파라미터가 제거되므로 호출부 수정 필요

### Phase 5: 로컬 문서 복사

```bash
# 분석/계획 문서
cp /home/ubuntu/cs479/3DML_Segmentation-Comp./Read/softgroup_integration_plan.md \
   /home/ubuntu/cs479/3DML_Merged/Read/
cp /home/ubuntu/cs479/3DML_Segmentation-Comp./Read/sampling_gap_analysis.md \
   /home/ubuntu/cs479/3DML_Merged/Read/

# cu126 설치 가이드 (GitHub의 cu124 버전 대체)
cp /home/ubuntu/cs479/3DML_Segmentation-Comp./SoftGroup_cu126_install.md \
   /home/ubuntu/cs479/3DML_Merged/
```

### Phase 6: 데이터셋 생성 검증

```bash
# 5개 씬만 먼저 돌려서 GLB로 확인
python dataset_tools/generate_synthetic_dataset.py \
  --source-root data/data/object_instance_segmentation \
  --output-dir data/synth_merged_debug \
  --splits train --variants-per-scene 1 \
  --debug-glb --debug-up-axis z --seed 42

# 분포 분석 (인스턴스당 점 수가 TA와 비슷한지 확인)
# mean ≈ 14,700, std ≈ 6,800 목표
```

### Phase 7: SoftGroup 학습 파이프라인 확인

GitHub에 이미 통합되어 있으므로 동작 여부만 확인.

```bash
cd /home/ubuntu/cs479/3DML_Merged

# npy → pth 변환 (소수 씬만)
python dataset_tools/npy_to_pth.py ...

# 단일 씬 overfit 테스트
python tools/train.py configs/softgroup/softgroup_nubzuki.yaml
```

shell 스크립트 (`train_nubzuki_softgroup_x2.sh`) 의 경로/설정이
새 환경과 일치하는지 확인 후 사용.

---

## 주의사항

- `data/` 폴더 (생성된 .npy 파일들)는 복사하지 않음. 새 클론에서 재생성.
- `assets_new/` (TA 테스트 데이터)도 복사 불필요. 분석용으로만 사용.
- `softgroup/ops/ops.*.so`는 cu124용이므로 Phase 3에서 반드시 재빌드.
  재빌드 후 `.so` 파일이 바뀌는데, git push 전에 `.gitignore`에 추가할지 결정 필요.
- GitHub의 `README_SoftGroup_cu124_install.md`는 cu124 기준 → cu126 환경에서는
  `SoftGroup_cu126_install.md` 사용.
- sparsehash 등 빌드 의존성은 `SoftGroup_cu126_install.md` 참고.

---

## 완료 기준

- [ ] `generate_synthetic_dataset.py` 실행 → 씬 생성 성공 + GLB로 layout mode 다양성 확인
- [ ] 인스턴스당 점 수 분포: mean ≈ 14,700, std ≈ 6,800
- [ ] `from softgroup.model import SoftGroup` import 성공
- [ ] SoftGroup 더미 forward pass 성공
- [ ] (선택) SoftGroup 단일 씬 overfit 테스트 성공
