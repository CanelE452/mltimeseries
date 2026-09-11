# 논문 종료 설계의 독자 질문 감사

[판정] `story-flow-audit` 적용. 질문①~⑥은 claim-gate 문서 Phase3에 모두 Figure/Table/Section으로 매핑했다. 이는 필요한 증거를 모두 확보했다는 뜻이 아니다. 논문 원고 제출 전에는 대화 맥락을 공유하지 않는 독립 리뷰가 추가로 필요하다.

## 구멍 — 앞 질문부터

- [미검증] ① 문제의 새로움: Time-PEFT 본문·부록 미확보로 정확한 중복 여부가 열려 있다. “local gain과 future utility가 다르다”의 개념적 최초성은 AFLoRA/기존 일반화 문헌 때문에 주장할 수 없다.
- [미검증] ② 실제 중요성: 역사적 반례는 있지만 독립 fresh source-period에서 발생 빈도·의사결정 손해가 측정되지 않았다.
- [확인] ③ 대조군 목적: F0는 적응의 절대 가치, HEAD는 기본 output readout, WIDE는 parameter budget, JOINT는 같은 head에 internal update 추가, Full FT는 상한을 보장하지 않는 제한 reference로 구분했다. [미검증] 새 데이터의 실제 parameter/선택 공정성 검증은 실행 전 남는다.
- [미검증] ④ 데이터/설정: 노출 ledger와 metadata-only fresh unit 확정이 필요하다. Stage A 초안에 기간·target·bootstrap 규약이 미정이라고 명시했다. 미정 항목을 plausible한 값으로 채우지 않았다.
- [확인] ⑤ 현재 관측: Study35의 WIDE 대비 이득과 F0 대비 손해를 함께 보고했다. [미검증] Stage A/B/C·최종검증은 미실행이므로 역사적 관측을 fresh replication으로 설명할 수 없다.
- [미검증] ⑥ 범위: 다른 backbone, 미노출 final family, pretraining overlap과 미래 update의 독립 재현이 남는다. UNKNOWN은 확인된 clean으로 바꾸지 않는다.

## 끊김과 최소 조치

- [판정] “WIDE를 이김→새 방법 필요” 연결을 제거하고 F0 및 chronological utility gate를 거치도록 했다. JOINT<F0면 방법 개발로 넘어가지 않는다.
- [판정] “현재 기여 양수→계속 업데이트” 연결을 가정하지 않는다. removal contrast C와 continuation contrast U를 먼저 정의하고 반례를 기술한다.
- [판정] “실험 번호가 많음→독립 근거 풍부” 연결을 제거한다. 같은 target/기간 공유를 노출 ledger 및 statistical unit에서 추적한다.
- [판정] “novelty 원문 미확보→NOVELTY_GATE_FAIL”로 바꾸지 않는다. UNKNOWN 때문에 진입을 보류하며 동일 주장의 직접 선행이 확인될 때만 fail을 선언한다.

## 불일치 검사

[확인] 핵심 수치는 evidence reconstruction에서 %F0와 원 loss를 구분했다. Study35 JOINT gain은 −4.304809%F0이고 loss increase는 +4.304809%F0로 부호가 반대다. 두 문장은 같은 사실이다.

[판정] novelty 문서의 chronological transfer change는 G(D)−G(V), Stage A의 transfer drop은 G(V)−G(D)다. **drop=−change**이며 이름과 부호를 섞지 않는다. 논문 표에는 gain drop 하나로 고정하고 수식을 캡션에 적는다.

[판정] TSFM(time-series foundation model), PEFT(parameter-efficient fine-tuning), F0(무적응 모델), HEAD/WIDE/JOINT(각 output-only/확장 output-only/head+LoRA procedure), V(선택), D(그 다음 개발 평가), C(현재 기여), U(미래 업데이트 효용)를 최종 원고에서 첫 등장에 정의한다.

[확인] 새 Stage A 결과 그림은 만들지 않았다. [판정] 질문에 답할 새로운 관측 없이 과거 그림을 fresh 결과처럼 배치하지 않는다. 원고 제출 단계에서는 이 문서의 미검증 항목을 실제 결과와 한계 문장으로 다시 감사한다.
