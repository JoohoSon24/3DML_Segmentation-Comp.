# 3DML Segmentation Competition 프로젝트 시작 가이드

## 프로젝트 개요

이 프로젝트는 **3D Point Cloud Segmentation Challenge**로, 3D 포인트 클라우드에서 "Nubzukis"라는 객체를 감지하고 세그멘테이션하는 인스턴스 세그멘테이션 작업입니다. 주어진 포인트 클라우드 데이터에서 배경(0)과 객체 인스턴스(1, 2, ...)를 분리하는 모델을 개발해야 합니다.

### 주요 목표
- 3D 포인트 클라우드 인스턴스 세그멘테이션 모델 구현
- MultiScan 벤치마크 데이터셋 사용
- 모델 파라미터 제한 (50M 이하)
- 추론 시간 제한 (300초)
- VRAM 제한 (24GB)

## 프로젝트 구조

```
3DML_Segmentation-Comp/
├── assets/
│   ├── sample.glb          # 객체 메쉬 데이터
│   └── test_0000.npy       # 테스트 데이터 예시
├── dataset.py               # 데이터셋 로더 (수정 불가)
├── evaluate.py              # 평가 스크립트 (수정 불가)
├── model.py                 # 모델 정의 (수정 필요)
├── visualize.py             # 시각화 스크립트 (수정 가능)
└── README.md                # 프로젝트 설명
```

## 환경 설정

### 1. Conda 환경 생성
```bash
conda create -n 3d-seg python=3.10 -y
conda activate 3d-seg
```

### 2. PyTorch 설치
```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

### 3. 기타 패키지 설치
```bash
pip install numpy scipy tqdm matplotlib
```

**참고**: 추가 라이브러리가 필요한 경우 Slack 채널을 통해 요청하세요. 승인된 라이브러리만 사용할 수 있습니다.

## 데이터셋 준비

1. **MultiScan 벤치마크 데이터셋**을 다운로드하세요.
   - 데이터셋 다운로드에 시간이 걸릴 수 있으니 미리 준비하세요.
   - 제공된 `sample.glb` 참조 객체를 사용하세요.

2. 데이터 구조 이해:
   - 입력: 포인트 클라우드 (색상 정보 포함, 9차원 특징)
   - 출력: 포인트별 인스턴스 레이블 (배경=0, 객체=1,2,...)

## 모델 개발

### 1. model.py 수정
- `DummyModel` 클래스를 실제 모델로 교체하세요.
- 모델 파라미터 수를 50M 이하로 유지하세요.
- 사전 학습된 네트워크 사용 금지.

### 2. 인터페이스 유지
```python
def initialize_model(ckpt_path: str, device: torch.device, ...) -> torch.nn.Module:
    # 모델 로드 및 초기화

def run_inference(model: nn.Module, features: torch.Tensor, ...) -> torch.Tensor:
    # 추론 수행, [B, N] 형태의 인스턴스 레이블 반환
```

### 3. 훈련 파이프라인 구현
- 자체 훈련 코드를 작성하세요.
- 제공된 `train`과 `val` 스플릿만 사용하세요.
- 추가 데이터셋 사용 금지.

## 평가 및 제출

### 1. 평가 실행
```bash
python evaluate.py --data_dir /path/to/data --ckpt_path /path/to/checkpoint
```

### 2. 제출 요구사항
- **중간 제출**: 4월 30일 (목) 23:59 KST
- **최종 제출**: 5월 9일 (토) 23:59 KST
- 제출처: KLMS
- `evaluate.py` 수정 금지
- CUDA 버전: 12.4 (기본값)

### 3. 제한사항 준수
- 모델 파라미터: 50M 이하
- 추론 시간: 300초 이하
- VRAM: 24GB 이하
- 사전 학습 모델 사용 금지
- 추가 데이터셋 사용 금지

## 시각화

`visualize.py`를 사용하여 결과를 시각화하세요. 필요에 따라 수정 가능합니다.

## 팁 및 주의사항

- **코드 인용**: 사용한 기존 코드, 모델, 논문을 명확히 인용하세요. 위반 시 0점 처리됩니다.
- **오픈소스 사용**: 승인된 오픈소스 구현만 사용하세요.
- **시간 관리**: 데이터 다운로드와 환경 설정에 시간을 할애하세요.
- **디버깅**: 제공된 평가 스크립트로 정기적으로 테스트하세요.

## 시작하기

1. 환경 설정 완료
2. 데이터셋 다운로드 및 준비
3. `model.py`에서 기본 모델 구현 시작
4. 훈련 및 검증 반복
5. 평가 스크립트로 성능 확인
6. 제출 준비

이 가이드를 따라 프로젝트를 시작하세요. 질문이 있으면 적절한 채널을 통해 문의하세요!