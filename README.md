# CMAAM: Cross-Modal Alignment-Aware Merging

멀티모달 대규모 언어 모델(MLLM)을 위한 크로스-모달 정렬 인식 머징 프레임워크

## 개요

CMAAM은 Vision-Language 모델의 모달리티 특성을 고려하여 **적응적으로 머징 계수를 계산**하는 새로운 방법론입니다.

### 핵심 아이디어

기존 모델 머징 방식은 모든 파라미터에 동일한 알파(α) 값을 적용합니다. 하지만 MLLM은 Vision Encoder, Cross-Modal Bridge, Language Model 등 서로 다른 역할을 하는 컴포넌트로 구성되어 있어, **컴포넌트별/레이어별로 다른 머징 전략이 필요**합니다.

CMAAM은 데이터 없이 파라미터 분석만으로 최적의 머징 계수를 자동 계산합니다.

## 주요 기능

### 1. 모달리티별 파라미터 분류
```
VISION_ENCODER  : 비전 인코더 (ViT, CLIP 등)
CROSS_MODAL_BRIDGE : mm_projector, adapter 등
LANGUAGE_MODEL  : LLM backbone
SHARED_EMBEDDING : embed_tokens, lm_head, norm
```

### 2. 크로스-모달 정렬 점수
- **PDA** (Parameter Distribution Alignment): 파라미터 분포 유사도
- **GFS** (Gradient Flow Similarity): 그래디언트 흐름 유사도
- **CMBC** (Cross-Modal Bridge Coherence): 브릿지 레이어 일관성

### 3. 적응적 알파 계산
```
α_layer = base_α × alignment_score × (1 - sensitivity_variance)
```
- 정렬 점수가 높으면 → 더 많이 머징
- 민감도 분산이 높으면 → 더 보수적으로 머징

### 4. 머징 전략
| 전략 | 설명 |
|------|------|
| `BASIC` | 단일 알파 적용 |
| `LAYERWISE` | 레이어별 다른 알파 |
| `COMPONENT` | 컴포넌트별 다른 알파 (Vision/Bridge/Language) |
| `FULL` | 레이어 + 컴포넌트 조합 |

## 설치

```bash
git clone https://github.com/yujuyeon0511/CMAAM.git
cd CMAAM
pip install -r requirements.txt
```

## 사용법

### CMAAM 머징 실행

```bash
python merge/cmaam/cmaam_merge.py \
    --source /path/to/source_model \
    --target /path/to/target_model \
    --output /path/to/output \
    --strategy full \
    --base-alpha 0.5 \
    --model-type qwen2vl \
    --analyze
```

**Arguments**:
- `--source`: 소스 모델 경로 (머징할 지식을 가진 모델)
- `--target`: 타겟 모델 경로 (기반이 되는 모델)
- `--output`: 출력 경로
- `--strategy`: 머징 전략 (`basic`, `layerwise`, `component`, `full`)
- `--base-alpha`: 기본 알파 값 (0.0 ~ 1.0)
- `--model-type`: 모델 타입 (`qwen2vl`, `llava`, `llava_onevision`, `cogvlm`, `mplugowl`)
- `--analyze`: 상세 분석 리포트 출력

### 지원 모델

| 모델 타입 | 지원 모델 |
|-----------|----------|
| `qwen2vl` | Qwen2-VL-7B-Instruct 등 |
| `llava` | LLaVA-1.5, LLaVA-1.6 |
| `llava_onevision` | LLaVA-OneVision |
| `cogvlm` | CogVLM, CogVLM2 |
| `mplugowl` | mPLUG-Owl, mPLUG-Owl2 |

---

## 한국어 VQA 파인튜닝 파이프라인

Qwen2-VL 모델의 한국어 VQA 성능 향상을 위한 LoRA 파인튜닝 파이프라인입니다.

### 데이터셋 정보

| 카테고리 | 샘플 수 | 비율 |
|----------|---------|------|
| document | 785,734 | 41.3% |
| general | 320,672 | 16.9% |
| latex | 234,300 | 12.3% |
| captioning | 166,912 | 8.8% |
| visualization | 138,154 | 7.3% |
| table_vqa | 119,923 | 6.3% |
| arxiv | 107,179 | 5.6% |
| math_problem | 18,280 | 1.0% |
| korean_problem | 9,240 | 0.5% |
| **총합** | **1,900,394** | 100% |

### 데이터 전처리

```bash
python scripts/prepare_korean_vqa_data.py \
    --source /path/to/korean_vlm_data \
    --output data/korean_vqa \
    --mode full  # tiny(10K), quick(100K), full(전체)
```

### 파인튜닝

```bash
python scripts/finetune_qwen2vl_korean.py \
    --model Qwen2-VL-7B-Instruct \
    --data data/korean_vqa \
    --output checkpoints/finetuned \
    --epochs 3 \
    --batch-size 1 \
    --grad-accum 16
```

**학습 설정**:
- LoRA rank: 64
- LoRA alpha: 128
- Learning rate: 2e-5
- Trainable params: 161M / 8.4B (1.91%)

### 평가

```bash
python scripts/evaluate_korean_vqa.py \
    --model checkpoints/finetuned/merged \
    --test-data data/korean_vqa/test.jsonl \
    --output eval_results/
```

### 전체 파이프라인 실행

```bash
bash scripts/run_korean_experiment.sh full
```

실행 순서:
1. 데이터 전처리
2. Qwen2-VL LoRA 파인튜닝
3. CMAAM 머징 (파인튜닝 모델 + 원본 모델)
4. 베이스라인 비교 (Linear α=0.3, 0.5, 0.7)
5. 평가

---

## 프로젝트 구조

```
CMAAM/
├── merge/
│   └── cmaam/
│       ├── __init__.py
│       ├── component_classifier.py  # 모달리티 컴포넌트 분류
│       ├── alignment_scorer.py      # 크로스-모달 정렬 점수
│       ├── sensitivity_analyzer.py  # 모달리티 민감도 분석
│       ├── adaptive_merger.py       # 적응적 머징 알고리즘
│       └── cmaam_merge.py           # CLI 실행 스크립트
├── scripts/
│   ├── prepare_korean_vqa_data.py   # 데이터 전처리
│   ├── finetune_qwen2vl_korean.py   # LoRA 파인튜닝
│   ├── evaluate_korean_vqa.py       # 평가
│   └── run_korean_experiment.sh     # 전체 파이프라인
└── requirements.txt
```

---

## 실험 결과

> 실험 진행 중...

### 비교 모델
| 모델 | 설명 |
|------|------|
| Qwen2-VL-7B-Instruct | 원본 (Baseline) |
| Fine-tuned | 한국어 VQA로 학습 |
| Linear (α=0.3, 0.5, 0.7) | 단순 선형 보간 |
| **CMAAM** | 적응적 크로스-모달 머징 |

### 평가 지표
- 한국어 VQA 정확도
- 영어 벤치마크 (영어 성능 유지 확인)

---

## 요구사항

- Python >= 3.10
- PyTorch >= 2.1.0
- transformers >= 4.36.2
- peft >= 0.7.0
- scipy >= 1.11.0

전체 목록은 `requirements.txt` 참조

---

## 라이선스

MIT License

## 감사의 글

이 프로젝트는 [AdaMMS](https://github.com/THUNLP-MT/AdaMMS)를 기반으로 확장되었습니다.
