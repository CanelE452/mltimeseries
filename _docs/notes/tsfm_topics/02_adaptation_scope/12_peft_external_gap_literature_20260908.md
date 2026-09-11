# 실제 원천 screen과 후속 방법의 신규성 경계

2026-09-08. [실행 계획](12_peft_external_gap_plan_20260908.md)의 학습 중에 확인한 원문이다. 아래 문헌 판단은 실제 유보 성능을 읽기 전의 것이다. 이것을 근거로 진행 중인 grid·기간·판정 문턱을 바꾸지 않는다.

## 이미 존재하는 설계

[TRACE v2](https://arxiv.org/html/2503.16991v2)의4장은 LoRA gate의 중요도가 다른 gate의 마스킹에 의존하는 문제를 다룬다. Validation에서 Monte Carlo 마스킹과 gate gradient를 모으고 중요도로 모듈을 남긴다. MOMENT용 prediction head 축소도 포함한다. 따라서 “중요한 LoRA 위치만 선택한다” 또는 “작은 출력층과 sparse LoRA를 결합한다”는 발상만으로 신규성을 주장할 수 없다. Chronos-2의 time/group attention 구분은 이 문헌과 다른 구조적 조건이지만, 구조 이름만 바꾼 중요도 선택은 충분한 차별점이 아니다. 출판본은 [Neurocomputing664,132098](https://doi.org/10.1016/j.neucom.2025.132098); 이번 방법 확인은 공개된2025-05-21 v2 원문에 근거한다.

[MixFT v1](https://arxiv.org/html/2603.02840v1)은 TSFM embedding의 Bayesian mixture로 fine-tuning 데이터를 잠재 하위 도메인으로 나누고 LoRA를 적합·결합한다. Fine-tuning과 평가 데이터가 하위 도메인을 공유한다는 가정과 여러 adapter의 메모리·학습 비용을 명시한다. 따라서 “시계열 상태별 adapter”나 “원 데이터셋 대신 잠재 regime으로 나눔”은 이미 가까운 설계다. 우리의 고정 target-prefix 적응과 이 논문의 관련 여러 데이터셋을 통한 zero-shot 전이는 다른 배포 문제이므로, 우열이나 재현으로 섞어 쓰지 않는다. 확인한 문서는2026-03-03 preprint v1이며 학회 채택을 주장하지 않는다.

[HiP-LoRA v1](https://arxiv.org/html/2604.17751v1)은 pretrained weight의 주요 singular subspace에 제한된 gain 수정을 허용하고, 양쪽 직교여공간의 low-rank update에 나머지 적응을 맡긴다. 주요 방향에 singular-value 가중 안정성 penalty를 적용한다. 따라서 “기존 예측 능력을 보호하며 여공간에서 적응한다”도 그 자체로 새 방법이 아니다. 논문은 LLM 중심 실험이며 TSFM에서의 효과는 직접 입증하지 않았다. TSFM에 적용한다는 이유만으로 방법론적 신규성이 자동으로 생기지 않는다. 확인한 문서는2026-04-20 preprint v1이다.

## 현재 탐색에 적용하는 판단

[When Do Foundation Models Pay Off?](https://arxiv.org/abs/2607.04919)는2026-07-06 preprint이며 짧은 데이터에서LoRA가악화할가능성과FM/고전모형선택을이미다룬다. 논문의훈련길이규칙을우리63개의겹치는origin에직접대입하면단위가맞지않는다. 우리train은64일1536hourly시점이며,단순히origin수가700보다작다고기존규칙의예외를발견했다고말하지않는다.

[TSFM-PEFT-Bench 공개artifact](https://huggingface.co/datasets/EvalData/tsfm-peft-bench)는구조·도메인·PEFT선택상호작용과selector평가를이미제공한다. README는NeurIPS2026D&B심사중이라고밝히며채택된논문으로취급하지않는다. [selector평가JSON](https://huggingface.co/datasets/EvalData/tsfm-peft-bench/resolve/main/results/selector_evaluation.json)의작은평가cell수와넓은구간을고려해야한다. 짧고의존적인temporalvalidation의선택regret는더좁은후속질문이될수있지만,문제진단만으로새PEFT방법이완성되는것은아니다.

우선 현재 고정 절차에서 내부 적응의 실용적 필요성을 확인한다. 양성이면 모듈 삭제·단순 입력/출력 대조로 그 효과를 설명하는 구체적 가설을 좁힌다. 새로운 방법은 그 가설이 예측하는 실패를 해결해야 하며, 기존 gate·mixture·보존 penalty와 matched information/selection/compute 대조가 필요하다.

음성이면 step0 선택, 짧은 V/C, 계절 이동, objective 차이, calibration 상호작용 중 어떤 원인이 가능한지 분리한다. 같은 유보 결과를 보고 새로운 방법을 조정했다면 그 데이터는 이후 개발 데이터로 기록한다. “기존 방법을 적용해 점수가 오른다”는 결과와 “기존 방법이 해결하지 못한 문제에 새 학습 규칙이 필요하다”는 주장을 구분한다.
