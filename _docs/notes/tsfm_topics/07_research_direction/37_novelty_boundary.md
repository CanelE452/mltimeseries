# Study37 — PEFT 논문을 닫기 위한 novelty 경계

[확인] 감사일: 2026-09-11. 조사 대상은 TSFM adaptation의 필요성과 chronological utility이며 새 adapter 제안이 아니다. 기존 [Study14 문헌 경계](14_next_method_literature_boundary_20260908.md), [Study32](32_future_utility_20260911.md), [Study35](35_head_convergence_20260911.md)를 읽고 아래 외부 원문과 대조했다. 기존 sealed 문서와 실험을 변경하지 않았다.

## 현재 판정

[판정] **`UNKNOWN` — 좁은 characterization 후보 A는 유지하되 novelty gate의 무조건 PASS는 보류한다.** 확인한 원문에서는 아래 세 estimand를 capacity-matched head/F0 대조와 함께 chronological replication으로 연결한 직접 동일 실험을 찾지 못했다. 이것은 신규성의 증명이 아니다. [확인] 2026-09-12 사용자 제공 Time-PEFT 20쪽 원문을 확보하여 아래 경계를 갱신했다. UNKNOWN은 더 이상 본문 미확보를 뜻하지 않으며, 좁은 질문의 독창성·충분한 독립 실증이 아직 확정되지 않았다는 뜻이다.

[판정] 넓은 주장인 “fine-tuning은 OOD에서 나빠질 수 있다”, “data signal로 적응 여부를 판단한다”, “adapter를 동적으로 동결/선택한다”, “validation 중요도를 측정한다”, “과거 오류로 다음 예측을 보정한다”는 단독 novelty 후보에서 제외한다. 아래 선행들과 직접 겹친다. 새 method 진입은 이 감사로 승인되지 않는다.

## 서로 바꾸어 부르면 안 되는 estimand

[판정] 아래는 이번 closure에서 사용할 **정의**다. 손실은 작을수록 좋으며 동일 평가 집합 Q의 F0 손실로 정규화한다. V와 다음 D에서 분모가 다르므로 각각의 상대효용을 계산한 뒤 차이를 정의한다. 수식의 정의 자체가 새 기여라는 주장은 하지 않는다.

```text
adaptation gain G_Q(a) = 100 * [L_Q(F0) - L_Q(a)] / L_Q(F0)
internal procedure advantage I_Q = 100 * [L_Q(WIDE) - L_Q(JOINT)] / L_Q(F0)
current contribution C_V(t) = 100 * [L_V(adapter off at t) - L_V(adapter on at t)] / L_V(F0)
future update utility U_D(t,b) = 100 * [L_D(freeze-LoRA branch at t+b)
                                     - L_D(continue-LoRA branch at t+b)] / L_D(F0)
chronological transfer change = G_D(selected-on-V model) - G_V(selected-on-V model)
```

[판정] C는 같은 checkpoint의 제거 개입이고 U는 공통 checkpoint/optimizer/RNG에서 시작하는 학습 절차 간 비교다. C>0은 현재 사용 중인 adapter가 도움이 된다는 뜻이며, 보존한 adapter를 더 업데이트하는 U>0과 동일하지 않다. U의 freeze branch에서 head를 계속 학습할지, clipping 집합을 어떻게 처리할지, STOP과도 비교할지는 사전 고정해야 한다. I>0이어도 G(JOINT)<0이면 “internal effect without adaptation value”다.

[확인] Study32의 S/D는 이미 보았던 V를 앞/뒤로 나눈 개발 자료이고, 추가 optimizer step의 미래와 달력상 미래를 동시에 엄격하게 봉인한 최종 시험이 아니다. MASKED_UPDATE는 clipping 대조지만 이후 head 궤적까지 고정하지 않는다. Study35는 새 개발 기간의 V 개선/D 손해를 기록했으며, backbone 원인과 분포 변화 원인을 확정하지 않았다. 수치 검산의 권위 있는 출력은 이번 작업의 `results/peft_paper_closure_v1/evidence_reconstruction.*`이며 이 문서는 재검산 결과를 대신하지 않는다.

## 검색 범위와 증거 수준

[확인] 검색은 필수 여섯 제목의 exact-title 검색, `time series foundation models fine tuning transferability ICLR 2026 ICML 2026 negative transfer`, 근접 문헌의 인용/저자 링크를 사용했다. PMLR, ICML, ICLR proceedings, ACL Anthology, OpenReview, 저자 arXiv/저자 GitHub를 확인했다. 외부 리뷰·자동 요약·검색엔진의 venue 추정을 근거로 채택하지 않았다.

[확인] LoRA+·ELF·ICFT는 공식 PMLR PDF를 메모리에서 읽었고, AdaLoRA·AFLoRA는 PDF의 방법/평가 부분을 확인했다. Representations·TimeTic·TRACE는 아래 명시한 arXiv 본문 버전을 읽었다. TATO는 공식 ICLR PDF §3–4를 확인했다. 원문 전체가 다운로드 가능했다는 사실과 모든 실험/코드를 독립 재현했다는 주장은 다르다. 이번에는 어떤 외부 실험도 재현하지 않았다.

[미검증] 검색의 완전성, 모든 최종 camera-ready와 공개 preprint 사이의 동일성, 비공개/미색인 최신 논문, 모든 부록의 동일 estimand 사용 여부는 보증하지 않는다. 이 문서는 체계적 문헌고찰의 전수 포함 주장이나 최초 발견 증명서가 아니다.

## 1. Time-PEFT

1. paper — [확인] *Time-PEFT: Temporal and Multichannel Complexity-Based Fine-Tuning for Time-Series Foundation Models*.
2. venue/year — [확인] ICML 2026 공식 [poster](https://icml.cc/virtual/2026/poster/61767), [공식 목록](https://icml.cc/Downloads/2026). [OpenReview 논문](https://openreview.net/forum?id=n8seTOinYs).
3. backbone — [확인] 원문 p.6 §5.1.2, p.15 §D.1.2: MOMENT small/base, UniTS, Chronos-bolt-small, TTM-r2. 본 실험의 Chronos/TTM은 encoder 사용, zero-shot은 원래 encoder-decoder 사용. Appendix A Fig.8은 원래 경로의 baseline도 제시한다. Chronos2와 동일 모델이 아니다.
4. adaptation target — [확인] 복잡한 target dataset의 forecasting 성능 개선.
5. PEFT mechanism — [확인] 저자 설명은 frequency top-k filtering adapter와 multichannel adapter를 제시한다.
6. adaptation 결정 시점 — [확인] p.4 Fig.4 및 §4는 dataset complexity와 fine-tuning 이득을 연결한다. p.12 Algorithm1은 주어진 target dataset의 fine-tuning 절차다. [미검증] 이 내용을 우리의 V→next-D update-utility selector와 동일시할 근거는 없다.
7. signal — [확인] spectral entropy 기반 temporal complexity, channel information flow 기반 multichannel complexity.
8. chronological future utility 직접 평가 — [확인] p.6 §5.1, p.16 §D.2, Tables2/6–9는 forecasting 성능과 horizon별 비교다. [판정] 이를 capacity-matched WIDE/F0 대조의 반복 future-period utility 검증으로 부르지 않는다.
9. contribution/update utility 구분 — [확인] p.12 Algorithm1은 LoRA·frequency·channel·head를 공동 최적화한다. [미검증] 현재 제거기여 C와 후속 업데이트가치 U를 분리한 동일 계약의 증거는 확인하지 못했다.
10. freezing/rank allocation — [확인] p.16 §D.1.3: LoRA rank8/scale32, q/k/v 또는 TTM fc1/fc2, 나머지 backbone frozen, forecast head 공동 학습. 우리의 native-head-frozen LoRA-only와 다르다.
11. TSFM 특화 요소 — [확인] 시간적 주파수 복잡도와 채널 의존성을 이용하는 adapter 설계.
12. 겹치는 주장 — [판정] “어떤 dataset에 PEFT가 필요한지 처음 연구”, “HeadOnly보다 LoRA가 좋은 경우를 처음 발견”, “temporal/multichannel complexity 기반 PEFT가 새로움”, “frequency/channel adapter 자체가 새 방향”은 금지한다. p.4 Fig.3–4, p.6 §5.1.3, p.7 Table2, p.12 Fig.8에서 이미 직접 다룬다.
13. 남는 gap — [판정] 현재 좁은 질문은 capacity-matched WIDE vs internal LoRA, F0 대비 실제 adaptation value, native-path/head 보존 LoRA-only, chronological future의 value 유지다. [미검증] 이 조합의 최초성은 아직 증명되지 않았다. p.16 §D.2.2 자체가 encoder-only 변경의 손해와 원래 경로 보존 결과를 논하므로 native path 중요성 자체도 최초 주장하지 않는다.

[확인] 2026-09-11 온라인 접근 실패 기록은 역사로 보존한다. 2026-09-12 로컬 Downloads의 `21386_Time_PEFT_Temporal_and_M.pdf`(20쪽)에서 원문을 확인했다. p.6 §5.1.3 및 p.16 §D.1.3은 zero-shot 외 baseline 모두 forecast head를 학습한다고 명시한다. p.12 Algorithm1도 같은 계약이다. 이 PDF는 저장소에 복사하거나 commit하지 않았다. [판정] 위 본문 근거를 이번 LoRA-only 수치 결과와 섞지 않으며 논문 초록의 개선율을 Chronos2 pinball 결과와 수치 비교하지 않는다.

## 2. LoRA+

1. paper — [확인] *LoRA+: Efficient Low Rank Adaptation of Large Models*.
2. venue/year — [확인] [ICML 2024 / PMLR](https://proceedings.mlr.press/v235/hayou24a.html), [공식 PDF](https://raw.githubusercontent.com/mlresearch/v235/main/assets/hayou24a/hayou24a.pdf).
3. backbone — [확인] RoBERTa/GPT-2/LLaMA 계열 및 이론·toy model.
4. adaptation target — [확인] NLP downstream task fine-tuning.
5. PEFT mechanism — [확인] LoRA A/B에 서로 다른 학습률과 고정 비율을 적용.
6. adaptation 결정 시점 — [확인] fine-tuning을 수행하는 설정에서 학습률을 정하며 F0/freeze 동적 선택기가 아니다.
7. signal — [확인] large-width scaling 분석과 hyperparameter 선택; 데이터 달력시간의 안정성 신호가 아니다.
8. chronological future utility 직접 평가 — [확인] 읽은 실험은 NLP/toy train/test와 수렴속도 평가이며 calendar V→D TSFM 평가가 아니다.
9. contribution/update utility 구분 — [확인] A/B feature-learning dynamics를 분석하지만 C/U fork contrast는 제시하지 않는다.
10. freezing/rank allocation — [확인] 핵심 방법은 LR 비대칭이며 adaptive freeze/rank 방식이 아니다.
11. TSFM 특화 요소 — [확인] 없다.
12. 겹치는 주장 — [판정] LoRA/head 차이를 곧바로 representation 필요성으로 해석하기 전에 optimization 설명을 다뤄야 한다.
13. 남는 gap — [판정] matched-head/F0와 달력시간 transfer를 연결하는 실증 질문은 별개다. 여기서 말하는 width-limit “stability”를 temporal utility stability로 인용하면 안 된다.

## 3. AdaLoRA

1. paper — [확인] *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning*.
2. venue/year — [확인] [ICLR 2023](https://openreview.net/forum?id=lq62uWRJjiY); [저자 PDF v2](https://arxiv.org/pdf/2303.10512v2)의 conference 표기와 본문 확인.
3. backbone — [확인] DeBERTaV3-base, BART-large 등 NLP pretrained models.
4. adaptation target — [확인] NLU, QA, generation downstream tasks.
5. PEFT mechanism — [확인] update를 PΛQ로 parameterize하고 importance에 따라 singular-value triplet 예산을 재배분.
6. adaptation 결정 시점 — [확인] fine-tuning 중 global budget schedule에 따라 반복 결정.
7. signal — [확인] gradient×weight sensitivity의 EMA와 변동성; §3.2 식(8)–(11).
8. chronological future utility 직접 평가 — [확인] 읽은 평가에는 TSFM next-calendar-period contrast가 없다.
9. contribution/update utility 구분 — [확인] 제거 시 손실 변화의 근사를 importance로 사용한다. 별도 continue/freeze fork의 미래 이득을 직접 분리하지 않는다.
10. freezing/rank allocation — [확인] adaptive rank allocation을 한다.
11. TSFM 특화 요소 — [확인] 없다.
12. 겹치는 주장 — [판정] importance 기반 rank/budget 배분 자체는 선행.
13. 남는 gap — [판정] importance가 현재 보존가치와 후속 업데이트가치를 언제 다르게 나타내는지 TSFM 시간축에서 검증할 여지는 남는다. AdaLoRA가 그런 보장을 했다고 과장해 반박하지 않는다.

## 4. AFLoRA

1. paper — [확인] *AFLoRA: Adaptive Freezing of Low Rank Adaptation in Parameter Efficient Fine-Tuning of Large Models*.
2. venue/year — [확인] [ACL 2024, Short Papers](https://aclanthology.org/2024.acl-short.16/), [PDF](https://aclanthology.org/2024.acl-short.16.pdf).
3. backbone — [확인] DeBERTaV3 실험; 해당 ACL PDF의 GLUE 평가를 기준으로 한다.
4. adaptation target — [확인] NLP downstream fine-tuning의 품질/계산량.
5. PEFT mechanism — [확인] low-rank projection 두 개와 feature transformation vector; projection을 점진적으로 동결.
6. adaptation 결정 시점 — [확인] 훈련 중 schedule과 freezing score로 결정.
7. signal — [확인] gradient magnitude와 그 smoothed uncertainty의 결합; §4 식(2)–(4). weight 크기만으로 동결하지 않는다.
8. chronological future utility 직접 평가 — [확인] 해당 PDF는 NLP benchmark/시간/연산량 평가이며 TSFM chronological V→D를 다루지 않는다.
9. contribution/update utility 구분 — [확인] 현재 큰 weight라도 변화가 작으면 동결 가능하다는 논리를 명시한다. C와 U를 같은 개념으로 취급한다고 주장하면 오독이다. [판정] 두 estimand의 불일치를 TSFM matched fork에서 계량하는 질문과는 구별된다.
10. freezing/rank allocation — [확인] adaptive freezing 있음.
11. TSFM 특화 요소 — [확인] 없다.
12. 겹치는 주장 — [판정] “유용한 adapter도 업데이트를 멈출 수 있다”와 동결 효율화의 넓은 동기는 이미 선행이다.
13. 남는 gap — [판정] calendar generalization/F0 포함 절대효용과 현재 contribution의 예측능력을 직접 검증해야만 추가 분석 기여가 된다.

## 5. Exploring Representations and Interventions

1. paper — [확인] *Exploring Representations and Interventions in Time Series Foundation Models*.
2. venue/year — [확인] [ICML 2025 / PMLR](https://proceedings.mlr.press/v267/wilinski25a.html); 검토 본문 [저자 arXiv v3](https://arxiv.org/html/2409.12915v3).
3. backbone — [확인] MOMENT, Chronos, Moirai.
4. adaptation target — [확인] 표현 중복성 분석, pruning, concept-informed forecasting intervention.
5. PEFT mechanism — [확인] CKA 기반 layer pruning과 latent concept steering; standard LoRA 필요성 비교가 핵심이 아니다.
6. adaptation 결정 시점 — [확인] 분석한 representation/concept에 따라 pruning 또는 inference-time steering.
7. signal — [확인] representation similarity, linearly represented trend/periodicity concepts.
8. chronological future utility 직접 평가 — [확인] forecasting/steering 평가는 있지만 검토 본문의 중심 실험은 V-selected internal PEFT의 next-D gain 보존 검증이 아니다.
9. contribution/update utility 구분 — [확인] C/U optimizer fork 비교를 제시하지 않는다.
10. freezing/rank allocation — [확인] layer pruning 있음; adaptive LoRA freezing/rank와 다르다.
11. TSFM 특화 요소 — [확인] trend/periodicity 개념과 시계열 latent representation.
12. 겹치는 주장 — [판정] TSFM에 주기/추세 정보가 표현되고 조작 가능하다는 주장은 신규하지 않다.
13. 남는 gap — [판정] representation의 존재/조작가능성과 target adaptation의 추가 미래효용은 다른 질문이다. fresh-head control 없이 “누락 정보를 복원했다”는 주장을 하지 않는다.

## 6. ELF / 이전 공개명의 AdapTS

1. paper — [확인] 공식 제목은 *Lightweight Online Adaption for Time Series Foundation Model Forecasts* (“Adaption” 표기).
2. venue/year — [확인] [ICML 2025 / PMLR](https://proceedings.mlr.press/v267/lee25ag.html), [공식 PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/lee25ag/lee25ag.pdf). [arXiv v2](https://arxiv.org/html/2502.12920v2)는 AdapTS 명칭을 사용한다.
3. backbone — [확인] TTM, TimesFM, VisionTS, Chronos, Moirai.
4. adaptation target — [확인] 배포 중 FM의 output forecast.
5. PEFT mechanism — [확인] backbone frozen; Fourier-domain linear forecaster의 online fitting과 FM/forecaster 가중 결합.
6. adaptation 결정 시점 — [확인] 매 M time steps 도착한 과거 label로 갱신하고 다음 forecast에 적용.
7. signal — [확인] 관측된 forecast loss; recent/full-history fast/slow exponential weighters와 merge.
8. chronological future utility 직접 평가 — [확인] **한다**. §3–4 rolling window와 label-arrival 계약을 명시한다.
9. contribution/update utility 구분 — [확인] frozen FM/online forecaster의 상대 예측성능을 평가한다. internal adapter on/off와 동일-state continue/freeze의 분해는 아니다.
10. freezing/rank allocation — [확인] backbone 항상 동결; LoRA rank allocation 없음.
11. TSFM 특화 요소 — [확인] delayed forecasting targets, frequency-domain efficient online correction.
12. 겹치는 주장 — [판정] “과거 오류를 이용해 미래 adaptation utility를 개선한다”와 lightweight temporal correction은 직접 선행.
13. 남는 gap — [판정] offline target-prefix LoRA의 내부 추가가치와 C/U 분석. 온라인 baseline을 비교하려면 동일 label 도착 정보를 허용해야 한다.

[판정] Study14의 AdapTS와 ELF를 서로 독립된 두 논문으로 세지 않는다. 공개 버전과 최종 proceedings의 명칭 차이를 기록하고 최종 학회명을 우선한다.

## 7. In-Context Fine-Tuning (ICFT)

1. paper — [확인] *In-Context Fine-Tuning for Time-Series Foundation Models*.
2. venue/year — [확인] [ICML 2025 / PMLR](https://proceedings.mlr.press/v267/faw25b.html), [공식 PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/faw25b/faw25b.pdf).
3. backbone — [확인] TimesFM.
4. adaptation target — [확인] target task에 맞는 in-context forecasting.
5. PEFT mechanism — [확인] 여러 example와 separator를 학습하도록 continued pretraining; 배포 시 예시를 제공한다. 내부 LoRA controller가 아니다.
6. adaptation 결정 시점 — [확인] upstream continued training 후 inference context 구성.
7. signal — [확인] 같은 task/dataset의 context examples.
8. chronological future utility 직접 평가 — [확인] forecasting benchmark의 미관측 future를 평가한다. [판정] 이것을 V→D adaptation-gain instability의 직접 시험과 같다고 세지 않는다.
9. contribution/update utility 구분 — [확인] 검토한 방법/실험에 C/U fork 없음.
10. freezing/rank allocation — [확인] 핵심 방법에 adaptive LoRA freeze/rank 없음.
11. TSFM 특화 요소 — [확인] time-series patches, examples의 autoregressive conditioning.
12. 겹치는 주장 — [판정] head/LoRA 외에도 target context를 이용하는 강한 적응 방식이 존재한다.
13. 남는 gap — [판정] fixed target-prefix 내부 PEFT의 절대/상대 utility 및 subsequent update 비교.

## 8. TATO

1. paper — [확인] *Adapt Data to Model: Adaptive Transformation Optimization for Domain-shared Time Series Foundation Models*.
2. venue/year — [확인] [ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/48c5226582f41254026748c7e35d4ac2-Abstract-Conference.html), [공식 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/48c5226582f41254026748c7e35d4ac2-Paper-Conference.pdf).
3. backbone — [확인] frozen LTM들; [미검증] 이 요약에서 모든 모델 variant 명칭을 재확인하지 않아 열거하지 않는다.
4. adaptation target — [확인] 입력/출력 변환 pipeline의 domain adaptation.
5. PEFT mechanism — [확인] context slicing, scale normalization, outlier correction; backbone weight update 없음.
6. adaptation 결정 시점 — [확인] D_history에서 pipeline을 선택하고 D_future에서 시험 (§3.1 정의2).
7. signal — [확인] historical forecasting loss, augmentations, Pareto filter와 weighted multi-indicator ranking.
8. chronological future utility 직접 평가 — [확인] historical→future 평가를 명시한다. [판정] temporal adaptation 평가가 없다는 주장을 해서는 안 된다.
9. contribution/update utility 구분 — [확인] internal adapter의 C/U fork는 방법의 대상이 아니다.
10. freezing/rank allocation — [확인] backbone frozen; LoRA allocation 없음.
11. TSFM 특화 요소 — [확인] 시간문맥·정규화·이상치 변환과 시간자료 augmentation.
12. 겹치는 주장 — [판정] history의 local fit과 future 사이 불일치를 이미 방법 설계에서 다룬다. 단순 “미래를 따로 봐야 한다”는 protocol novelty는 약하다.
13. 남는 gap — [판정] 동일 parameter/budget의 WIDE/JOINT와 current contribution/continued update의 정량적 연결. 외부 adaptation만으로 충분할 수 있다는 경쟁 설명을 강화한다.

## 9. TimeTic

1. paper — [확인] *Estimating Time Series Foundation Model Transferability via In-Context Learning*.
2. venue/year — [확인] [arXiv 2025 v1](https://arxiv.org/html/2509.23695v1). [미검증] [ICLR 2026 제출본](https://openreview.net/forum?id=P6HdPrHiQI)은 검색된 PDF에 under-review로 표시되며 최종 채택 상태는 UNKNOWN. accepted conference로 쓰지 않는다.
3. backbone — [확인] 10 TSFM benchmark; selector는 TabPFN.
4. adaptation target — [확인] 새 target dataset에서 fine-tuning할 backbone을 고르는 model selection.
5. PEFT mechanism — [확인] 검토 v1 Appendix B.3은 모델 전체를 1 epoch fine-tune하여 ground-truth ranking을 만든다. PEFT controller가 아니다.
6. adaptation 결정 시점 — [확인] target 모델 fine-tuning 전; source fine-tuning 관측을 context table로 사용.
7. signal — [확인] dataset meta-features, layer entropy profile, 과거 model-data performance.
8. chronological future utility 직접 평가 — [확인] downstream test 성능/ranking을 예측한다. [판정] 동일 모델의 V-local gain→next-D gain과 같은 estimand가 아니다.
9. contribution/update utility 구분 — [확인] 검토 v1에는 adapter on/off 대 continuation/freeze contrast가 없다.
10. freezing/rank allocation — [확인] 없음.
11. TSFM 특화 요소 — [확인] time-series characteristics와 TSFM layer entropy profile.
12. 겹치는 주장 — [판정] “적응 후 이득을 데이터/모델 신호로 미리 예측한다”는 넓은 B claim과 겹친다.
13. 남는 gap — [판정] 한 backbone에서 F0/LoRA/freeze action의 시간별 utility를 구분하고 target-label 노출 없이 예측하는 문제.

## 10. TRACE

1. paper — [확인] *TRACE: Time Series Parameter Efficient Fine-Tuning*.
2. venue/year — [확인] [저자 arXiv v3 (2025-11-21)](https://arxiv.org/html/2503.16991v3)는 Neurocomputing을 명시하고 [출판사 논문 레코드](https://www.sciencedirect.com/science/article/abs/pii/S0925231225027705)가 존재한다. [미검증] 출판사 full metadata/권호연도를 확보하지 못했으므로 2026 학회 논문으로 표기하지 않는다.
3. backbone — [확인] MOMENT-base 중심; NLP 확장도 보고한다.
4. adaptation target — [확인] forecasting/anomaly detection downstream adaptation.
5. PEFT mechanism — [확인] reconstructed head와 Gated DSIC의 LoRA module masking.
6. adaptation 결정 시점 — [확인] train α steps 후 validation 중요도 측정과 masking 반복.
7. signal — [확인] random masking 조건에서 gate gradient importance를 Monte Carlo 평균; §4.3 식(11)–(12).
8. chronological future utility 직접 평가 — [확인] forecast test 평가가 있다. [미검증] strict independent rolling chronological utility characterization은 본문에서 직접 확인되지 않는다.
9. contribution/update utility 구분 — [확인] masking 맥락에 맞춘 importance 추정을 명시한다. [판정] learned weight 제거와 future continue/freeze의 효용 차이를 동일-state 분기로 시험하는 것은 다른 질문이다.
10. freezing/rank allocation — [확인] 동적/reversible LoRA module 선택을 제안한다.
11. TSFM 특화 요소 — [확인] long-horizon head 크기와 temporal task adaptation.
12. 겹치는 주장 — [판정] validation contribution 추정, LoRA 중요도, head 용량 문제를 새롭게 제기했다고 하면 중복 위험이 높다.
13. 남는 gap — [판정] matched WIDE/F0 대조와 next-period C/U 불일치의 재현. TRACE importance를 future-utility guarantee라고 해석하거나 반박하면 안 된다.

## 11. Fine-Tuning Can Distort Pretrained Features

1. paper — [확인] *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution*.
2. venue/year — [확인] [ICLR 2022](https://openreview.net/forum?id=UYneFzXSJWh), [저자 논문](https://arxiv.org/abs/2202.10054).
3. backbone — [확인] pretrained vision models와 two-layer linear theoretical model.
4. adaptation target — [확인] downstream classification의 ID/OOD generalization.
5. PEFT mechanism — [확인] linear probing, full fine-tuning, LP-FT 비교; LoRA 구조 제안이 아니다.
6. adaptation 결정 시점 — [확인] downstream training과 LP→FT 순서 설계.
7. signal — [확인] head 초기화와 feature distortion 이론/실험.
8. chronological future utility 직접 평가 — [확인] OOD 평가를 한다. [판정] chronological TSFM PEFT로 범위를 대체할 수 없다.
9. contribution/update utility 구분 — [확인] 논문의 주요 질문은 joint head/backbone adaptation의 OOD 효과다. [미검증] 모든 부록의 fork-equivalent 분석을 이 감사에서 전수 배제한 것은 아니다.
10. freezing/rank allocation — [확인] LP 단계에서 backbone freeze; adaptive LoRA rank는 없다.
11. TSFM 특화 요소 — [확인] 없다.
12. 겹치는 주장 — [판정] ID/local 개선과 OOD 손해의 공존, head/backbone 공동 최적화 대안 설명은 기존 선행이다.
13. 남는 gap — [판정] TSFM의 capacity-matched output/F0 및 chronology-aware update utility 실증. 단지 OOD 문제의 TSFM 예시 하나만으로는 충분한 신규성이 약하다.

## 후보 A/B/C와 중복 위험의 최종 경계

[판정] **Primary A — CHARACTERIZATION / ANALYSIS (조건부)**. 완성 후 검증할 claim: “Internal LoRA can outperform capacity-matched output adaptation, but local adaptation gains need not transfer chronologically; current adapter contribution and future update utility are distinct.” 현재는 “development observations motivate testing whether...” 수준으로 낮춘다. 여러 독립 source/period와 엄격한 F0 대조로 **정량적 조건·빈도·반례**를 제공해야 분석 논문이 된다.

[판정] **B — 보류**. AFLoRA/AdaLoRA/TRACE의 학습 제어, Time-PEFT/TimeTic의 성능 proxy, ELF/TATO의 시간적 적응을 넘어서는 action utility 검증이 필요하다. 지금 signal의 예측력이나 새 adapter의 필요성은 확보하지 않았다. 기존 importance를 다르게 이름 붙여 controller를 만들지 않는다.

[판정] **C — 보조 contribution**. “미래를 따로 평가하라”만으로는 ELF/TATO와 일반 OOD 연구 대비 충분하지 않다. 동일 규약으로 여러 모델/자료에서 기존 PEFT ranking·선택 결론이 실제로 바뀌는 증거가 있어야 독립 evaluation paper 후보로 올린다.

[판정] `NOVELTY_GATE_FAIL` 전환 조건: Time-PEFT 또는 다른 근접 원문이 (i) capacity-matched head/F0, (ii) V-local versus next-period adaptation gain, (iii) same-state adapter contribution versus subsequent update/freeze utility를 **실질적으로 같은 질문으로 직접 검증**했고, 이번 연구가 독립 범위 확장 이상의 추가 기여를 제시하지 못하는 경우다. 표면적으로 같은 “stability/importance/future” 단어만으로 fail을 결정하지 않는다.

[확인] Time-PEFT 원문 확보·주요 baseline/학습 계약 확인은 2026-09-12 닫았다. [미검증] 남는 항목은 좁은 실증 조합의 충분한 독창성, TimeTic 채택 상태, TRACE 최종 출판 metadata와 검토본의 일치다. [판정] Time-PEFT의 complexity·HeadOnly/LoRA·native architecture 논의를 반영해 claim을 좁히며, 이번 개발 진단의 성공 여부를 novelty PASS로 대체하지 않는다.
