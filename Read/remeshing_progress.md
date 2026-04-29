# Remeshing Progress

## 목표

`assets/sample.glb` (Nubzuki 레퍼런스 오브젝트)를 MultiScan 데이터셋처럼 균일하고 정돈된 메시로 변환하는 것.
원본은 3D 스캔 기반이라 삼각형 크기가 불균일하고 구멍(open boundary)이 많음.

---

## 원본 메시 (`sample.glb`)

- Vertices: **11,829**
- Faces: **19,426** (삼각형)
- UV 텍스처 (PBR material)
- `is_watertight`: False — boundary edge 4,082개 (구멍 있는 open mesh)

---

## 흐름 및 시도 과정

### 1단계: pymeshlab 삼각형 리메싱 시도 → `sample_remeshed.glb`

**시도한 것:**
pymeshlab의 isotropic explicit remeshing으로 삼각형 메시를 균일하게 재배치.
버텍스 수를 원본과 비슷하게 유지하기 위해 bounding box 대각선 길이로 target edge length를 계산해서 입력.

**변환 과정:**
```
sample.glb → (trimesh로 OBJ 변환) → pymeshlab isotropic remesh → OBJ → GLB
```

**결과:**
- Vertices: **15,396** (원본 대비 +30%)
- Mesh type: 삼각형 유지
- 색 전달 없음 (UV 좌표가 리메싱 과정에서 소실됨)

**문제:**
- 사각형(quad) 메시가 아니라 삼각형이라 MultiScan처럼 격자 형태가 안 됨
- 색이 없음

---

### 2단계: Instant Meshes로 quad 리메싱 시도

quad 메시를 만들기 위해 Instant Meshes를 도입.
Instant Meshes는 prebuilt 바이너리가 없어서 소스 빌드 필요.

**빌드 이슈:**
- bundled TBB가 최신 GCC와 호환 안 됨 → `cmake . -DCMAKE_CXX_FLAGS="-Wno-changes-meaning"` 로 해결
- OpenGL 헤더 없음 → `libgl1-mesa-dev`, `libxxf86vm-dev` 설치
- 바이너리 경로: `third_party/instant-meshes/instant-meshes/Instant Meshes` (파일명에 공백 있음)

**Instant Meshes 내부 동작:**
- CLI batch mode로 디스플레이 없이 실행 가능
- 입력 vertex 수의 1/4을 `--vertices`에 넘겨야 원하는 수가 나옴
  (내부적으로 Step 9에서 quad subdivision을 한 번 더 해서 vertex가 ~4배 늘어나기 때문)

---

### 2-1: hole closing 후 quad 리메싱 → `sample_quad.glb`

**시도한 것:**
원본에 구멍이 많아서 Instant Meshes가 경계를 처리 못하는 문제를 해결하기 위해
pymeshlab `meshing_close_holes(maxholesize=200)`로 먼저 구멍을 채운 뒤 quad 리메싱.

**변환 과정:**
```
sample.glb
→ trimesh로 OBJ 변환
→ pymeshlab hole closing (boundary loop를 flat 삼각형으로 막음)
→ OBJ 저장
→ Instant Meshes quad remesh (--vertices = 11829 // 4 = 2957)
→ OBJ 출력
→ 원본에서 UV 샘플링한 색을 KDTree nearest-neighbor로 전달
→ GLB 저장
```

**결과:**
- Vertices: **~12,635** (원본 대비 +7%)
- Mesh type: pure quad
- 색 전달 완료

**문제:**
- `meshing_close_holes`는 구멍의 경계 루프(boundary loop)를 평평한 삼각형으로 막는 방식
- 공간적 거리를 고려하지 않아서, 다리 사이 바닥처럼 큰 구멍은 멀리 떨어진 부분끼리 연결되어 이상하게 보임
- 머리 내부, 다리 사이 등에 납작한 막이 생김

---

### 2-2: hole closing 없이 quad 리메싱 → `sample_quad_nofill.glb`

**시도한 것:**
구멍을 억지로 채우지 않고 원본 그대로 Instant Meshes에 넣어봄.

**변환 과정:**
```
sample.glb
→ trimesh로 OBJ 변환
→ Instant Meshes quad remesh (--vertices = 2957)
→ 색 전달 (KDTree)
→ GLB 저장
```

**결과:**
- Vertices: **~42,330** (원본 대비 +258%)
- Mesh type: pure quad
- 구멍 부분에 경계 tear 발생 (찢긴 형태)

**문제:**
- 구멍 없이 넣으면 Instant Meshes가 경계 처리를 제대로 못해서 vertex가 많이 늘어남
- 형태가 찢겨 보임

---

### 3단계: Poisson surface reconstruction으로 전처리 개선

hole closing의 문제(flat cap, 원거리 연결)를 해결하기 위해
노멀 기반 Poisson reconstruction으로 대체.

**Poisson reconstruction 원리:**
- 버텍스의 노멀 정보를 기반으로 포아송 방정식을 풀어서 표면을 새로 추정
- 구멍을 주변 곡률에 자연스럽게 맞춰 채움
- 결과가 watertight mesh (완전히 닫힌 메시)

**중간 확인용 결과 → `sample_poisson.glb` (시각화용)**
Poisson reconstruction만 적용하고 quad 변환 없이 GLB로 저장해서 결과 확인.
→ Vertices: **9,713** / Faces: **19,426** (원본과 같은 face 수로 decimation)

---

### 3-1: Poisson + decimation + quad 리메싱 → `sample_quad_poisson.glb`

**시도한 것:**
Poisson으로 watertight mesh 만든 뒤 decimation으로 vertex 수 줄이고, Instant Meshes로 quad 변환.

**변환 과정:**
```
sample.glb
→ trimesh로 OBJ 변환
→ pymeshlab Screened Poisson Reconstruction (depth=8)
  : 노멀 기반 표면 재추정, watertight mesh 생성 (32,803 vertices로 늘어남)
→ pymeshlab quadric edge collapse decimation (target face = 19,426)
  : vertex를 9,713으로 줄임
→ trimesh.Trimesh(vertex_matrix, face_matrix)로 직접 조립 (OBJ 경유 시 로드 실패 이슈 있음)
→ OBJ로 저장
→ Instant Meshes quad remesh (--vertices = 2957)
→ 색 전달 (원본 UV 샘플링 → KDTree nearest-neighbor)
→ GLB 저장
```

**결과:**
- Vertices: **~47,455** (원본 대비 +301%)
- Mesh type: pure quad
- Instant Meshes에서 "1 holes" — 이전 quad.glb(구멍 다수)보다 훨씬 깔끔
- 색 전달 완료

---

## 결과 파일 요약

| 파일 | Vertices | 원본 대비 | Faces | 메시 타입 | 특이사항 |
|---|---|---|---|---|---|
| `sample.glb` | 11,829 | 기준 | 19,426 | 삼각형 | 원본, open mesh |
| `sample_remeshed.glb` | 15,396 | +30% | — | 삼각형 | pymeshlab isotropic remesh, 색 없음 |
| `sample_filled.glb` | 9,713 | -18% | 19,426 | 삼각형 | Poisson + decimation만 (시각화용) |
| `sample_quad.glb` | ~12,635 | +7% | ~10,669 | pure quad | hole closing + Instant Meshes, 색 있음 |
| `sample_quad_nofill.glb` | ~42,330 | +258% | ~21,604 | pure quad | hole closing 없이 Instant Meshes, 찢김 |
| `sample_quad_poisson.glb` | ~47,455 | +301% | ~23,904 | pure quad | Poisson + decimate + Instant Meshes, 색 있음 |

---

## 기술 메모

- **색 전달 방법**: 리메싱 후 UV 좌표가 사라지므로, 원본 버텍스 위치에서 UV 텍스처를 샘플링해 색을 구한 뒤 새 메시 버텍스에 KDTree nearest-neighbor로 전달. PBR material은 `material.baseColorTexture` 사용 (`material.image` 아님).
- **pymeshlab OBJ 로드 이슈**: pymeshlab 결과를 OBJ로 저장하고 trimesh로 다시 로드하면 vertex가 크게 줄거나 손실되는 경우 있음. `vertex_matrix()`, `face_matrix()`로 직접 추출해서 `trimesh.Trimesh()`로 조립하는 게 안전.
- **Instant Meshes vertex 수 보정**: 내부 quad subdivision(Step 9)이 vertex를 ~4배 늘리므로 `--vertices`에 목표값 // 4를 넘겨야 함.
- **Poisson depth**: `depth=8`이 디테일과 속도의 균형점. 높일수록 디테일 살아나지만 vertex 수 증가.
