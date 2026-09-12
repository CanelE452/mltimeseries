# Study38 — native-path LoRA-only diagnostic

[판정] 전체 CASE C. 이미 노출된 BMRA/Jena Study35 개발 분할의 진단이며 independent/final test가 아니다.

[확인] LoRA-only와 JOINT가 모두 F0보다 나쁜 cell/view는 8/8이다. LoRA-only는 JOINT보다 7/8에서 좋다. BMRA seed30001 S180에서는 JOINT가 LoRA-only보다 7.461550%F0 좋으며, 나머지7개에서는 LoRA-only가 더 좋다.

[추정] 추가 residual MLP만으로 Study35 D 손해를 설명하기 어렵다. 이번 조건에서는 native-head-frozen LoRA-only도 F0 대비 adaptation value를 얻지 못했다. 새 adapter를 만들기보다 fresh chronological 검증을 우선한다.

[확인] 아래 raw D loss는 작은 값이 좋다. S180/L720은 같은 720-step trajectory에서 V로 선택한 두 view이며 독립 반복이 아니다.

```text
dataset seed   view   F0          HEAD        WIDE        LORA_ONLY   JOINT       CASE
bmra    30000 L720  0.415325056 0.476327123 0.472781197 0.429246262 0.441710879  C
bmra    30000 S180  0.415325056 0.476327123 0.472781197 0.429246262 0.441710879  C
bmra    30001 L720  0.415325056 0.496245817 0.483261077 0.429430633 0.435020436  C
bmra    30001 S180  0.415325056 0.484546948 0.491938820 0.464995071 0.434005385  C
jena    30000 L720  0.475301237 0.540377190 0.517708427 0.485082755 0.489484614  C
jena    30000 S180  0.475301237 0.520608416 0.517708427 0.485082755 0.489484614  C
jena    30001 L720  0.475301237 0.560355164 0.538625757 0.486854695 0.490225416  C
jena    30001 S180  0.475301237 0.503809480 0.538625757 0.486854695 0.490225416  C
```

[확인] 아래 대비는 모두 %F0. G는 F0 대비, I는 WIDE 대비, H_EFFECT는 100(LoRA-only−JOINT)/F0이며 양수면 JOINT가 더 좋다.

```text
dataset seed   view   G_LORA    G_JOINT   I_LORA    I_JOINT   H_EFFECT
bmra    30000 L720    -3.3519   -6.3531  +10.4821   +7.4810   -3.0012
bmra    30000 S180    -3.3519   -6.3531  +10.4821   +7.4810   -3.0012
bmra    30001 L720    -3.3963   -4.7422  +12.9610  +11.6152   -1.3459
bmra    30001 S180   -11.9593   -4.4978   +6.4874  +13.9489   +7.4615
jena    30000 L720    -2.0580   -2.9841   +6.8642   +5.9381   -0.9261
jena    30000 S180    -2.0580   -2.9841   +6.8642   +5.9381   -0.9261
jena    30001 L720    -2.4308   -3.1399  +10.8923  +10.1831   -0.7092
jena    30001 S180    -2.4308   -3.1399  +10.8923  +10.1831   -0.7092
```

[판정] CASE A는 LoRA-only가 F0보다 좋고 JOINT 이상, B는 JOINT가 F0와 LoRA-only보다 좋음, C는 둘 다 F0보다 나쁨을 뜻한다. D는 앞 조건에 해당하지 않으면서 다섯 arm 모두 기존0.25%F0 band 안인 경우다. 어느 조건에도 맞지 않으면 UNRESOLVED로 남긴다. 서로 다른 CASE가 나타나면 전체 E로 표기한다.

[추정] A cell은 추가 residual MLP가 필요하지 않았거나 공동 적응이 불리했을 가능성과 양립한다. C cell의 손해는 추가 head만으로 설명하기 어렵다. 이 비교는 학습 절차 간 차이이며 representation 복원·정보 생성·표현 부족을 직접 측정하지 않는다. LORA_ONLY의 학습 파라미터 수는 WIDE/JOINT보다 적다.

[확인] 기존 Study35 L720 재검산 평균: JOINT−HEAD 개선 +12.134632%F0, JOINT−WIDE 개선 +8.804323%F0, JOINT−F0 개선 −4.304809%F0. 기존 결과 재학습은 없다.

[확인] Step0 F0 max error=0.0; 실제 학습 파라미터=[1179648]; 96 attention LoRA modules/192 tensors. Native head와 backbone 원래 가중치 frozen, residual probe 없음. 학습 후 frozen hash 및 선택 checkpoint 재복원/예측 일치 검사를 통과했다.

[확인] 기존 Study35 소스·입력 해시 59개, closure provenance 486개, 기존 raw D 32행 및 V-only 선택 재검산 통과. 새 plan SHA256 `40226cc6e1ce7b293140faba3e08663cfb075c3fcae631b0040c4594d76747dd`; selection SHA256 `efa517f76fc86dff6d0b304b2056e61dc0f57d894de325cd1e8567f9f3514e00`. metric/target/scale/quantile/origin과 native F0 D 예측이 기존 증거와 일치한다.

[확인] LR [1e-5,3e-5,1e-4,3e-4]는 요청한 네 default와 기존 HEAD/WIDE numeric grid를 사용했다. Study35 JOINT LoRA rate는 [1e-5,3e-5,3e-5,1e-4]였으므로 동일한 JOINT grid라고 주장하지 않는다. D를 보고 grid나 threshold를 변경하지 않았다.

[확인] 실행: fit 16회, 선택 D forecast 8회, zero-update smoke 4회; guard 28회 정상 종료. 최초 plan부터 마지막 forecast까지 wall 2957.42s(대기·정리 포함, 이전 소스 감사·후속 보고 제외), 학습/평가 controller wall 2419.35s, fit worker wall 합 2098.43s, 학습 trajectory 합 2045.13s.

[확인] Peak CUDA allocated 946.5MiB / reserved 964.0MiB; sampled device 전체 메모리 2053.0MiB. Runtime 최소 여유 RAM 12.75GiB, commit 9.34GiB. 장치 sampled memory와 PyTorch allocator peak는 서로 다른 측정이다.

[확인] 최초 준비에서 로그 폴더 생성 누락으로 GPU 실행 전 FileNotFoundError가 발생했다. 최초 plan과 traceback을 보존하고 폴더 생성 순서만 수정한 뒤 재봉인했다. 메모리 기준을 낮추지 않았으며, 사용자 승인으로 부모가 종료된 watcher/helper와 미사용 자동화 도우미, 오래된 유휴 Claude CLI를 정리했다. 학습·원천·sealed 결과 파일은 삭제하지 않았다.

[판정] Fresh Stage A: BDG2 Bull Office 2016-01-01~2016-08-10 READY(개발 후보); Household post-P1 BLOCKED(48h target 결측 창). Household는 성능을 보지 않은 origin exclusion/gap 또는 split 정책을 먼저 정해야 한다. 두 후보 pretraining overlap은 UNKNOWN. Fresh 학습/forecast는0회다.

[확인] Time-PEFT 원문 p.6 §5.1.3, p.12 Algorithm1, p.16 §D.1.3에서 forecast-head 공동 학습을 확인했다. p.16 §D.2.2는 native architecture 보존 여부와 손해 차이도 논한다. 해당 경계는 별도 Study37 문서에 갱신했으며 PDF는 commit하지 않는다.

[판정] 새 adapter 개발이나 READY_FOR_PAPER 승격은 하지 않는다. 결과를 검토한 뒤 fresh F0/HEAD/WIDE/LORA_ONLY/JOINT 계약을 별도로 봉인해야 한다. 이번 실행은 main commit/push로 끝낸다.

[확인] [결과와 그림](../../../../results/peft_lora_only_diagnostic_v1/STATUS.md) · [fresh manifest](../../../../results/peft_paper_closure_v1/fresh_stage_a_candidate_manifest.md) · [선행 경계](37_novelty_boundary.md)
