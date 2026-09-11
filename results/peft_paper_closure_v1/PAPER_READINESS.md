# PEFT paper readiness — 2026-09-11

[판정] **MORE_INDEPENDENT_EVIDENCE_REQUIRED**

[판정] Primary는 **A — characterization / analysis**다. 이번 작업은 기존 근거를 재구성하고 논문 종료 규칙을 고정한 감사 단계까지 수행했다. 논문 준비 완료, 현상 재현 실패, novelty 중복 확정 중 어느 것도 선언하지 않는다. **novelty UNKNOWN과 인증된 fresh/final manifest 부재 때문에 Stage A를 실행하지 않았다.** 이는 사용자 승인 부족이 아니라 요청한 상류 gate의 미충족이다.

## 이번에 확인한 것

[확인] origin fetch 후 최신 PEFT 연구 branch는 `origin/peft-method-pilot-screen-v1`, head는 `33ede9bf9694c5199f7f8cfce6c8b34551b451f8`이다. 현재 `peft-paper-closure-v1`이 이미 그 commit에 있어 재사용했다. `peftpaperclosurev1`도 동일 commit이며 중복 branch를 더 만들지 않았다. main의 초기 README로 연구 상태를 판단하지 않았다.

[확인] 독립 CPU 재계산은 **936개 수치 대조, 486개 입력 hash 불변, EVIDENCE_MISMATCH 0**으로 통과했다. 최종 parent 실행 exit0, 계산 wall-clock **3.910906초**다. 문서 반올림 허용 폭과 원시 수치 절대 오차1e−10을 별도로 적용했다. [원시 비교와 범위](evidence_reconstruction.md), [모든 수치·hash](evidence_reconstruction.json).

- [확인] Study20 Bike LoRA 추가 이득 +3.953447%F0, Study26 동일 residual MLP 뒤 추가 이득 +4.170706%F0.
- [확인] Study30 P1 Bike FULL90은 WIDE/JOINT trainable parameter 각1,768,949개, JOINT 추가 이득 **+4.778926%F0**.
- [확인] Study32의24개 의존된 분기 중 현재 contribution 양수21개, 그중16개에서 future update utility가 음수였다. 이 분기를 독립 dataset24개라고 세지 않는다.
- [확인] Study35 JOINT는 WIDE보다 **8.804323%F0** 좋지만 F0보다 **4.304809%F0** 나빴다. 두 원천·두 seed 네 cell 모두 F0 손해다. 별도 CSV 집계와 원시 NPZ 재계산의 평균 차이는1.78e−15였다.
- [확인] Study36 sign conjugacy, latest-state quadratic Bayes predictor와 one-step MSE0.46을 CPU 모집단 계산으로 재확인했다. 현재 DGP의 GPU 실험은 재개하지 않았다.

[미검증] 이 재구성은 모든 부차 지표·모델 선택 기회·bootstrap·원인 기전을 전수 감사한 것은 아니다. Study32 off prediction은 원시 실행 JSON에 의존하고, Hospital individual/shuffled는 elementary CSV 대조다. Hospital F0/LoRA는 원시 prediction도 재계산했다. 상세 한계를 evidence report에 명시했다.

## 약해진 주장과 남는 primary claim

[판정] “WIDE보다 좋으니 adaptation이 유용하다”는 주장은 F0 대조를 빠뜨린다. “내부 표현에 정보가 없었다”는 설명은 현재 실험으로 입증되지 않았다. 많은 study 번호·seed·겹친 origin은 독립 재현 수를 늘리지 않는다.

[확인] [11편 novelty audit](../../_docs/notes/tsfm_topics/07_research_direction/37_novelty_boundary.md)에서 AFLoRA의 보존/업데이트 구분, ELF·TATO의 chronological adaptation 평가, TRACE의 validation importance/head capacity, Time-PEFT의 complexity-based fine-tuning이 근접 선행으로 확인됐다. [판정] 개념적 “현재 useful≠추가 update 필요”나 “미래를 따로 평가” 자체의 최초성을 주장하지 않는다.

[미검증] Time-PEFT 공식 채택·초록·저자 구현은 확인했지만 공식 본문/부록은 여러 경로에서403으로 확보하지 못했다. 따라서 정확히 같은 C/U 실험의 부재를 확정할 수 없으며 novelty gate는 **UNKNOWN**이다. 접근 실패는 논문 중복의 증거가 아니다.

[판정] 검증할 primary claim 한 문장: **“TSFM에서 internal LoRA가 동일 용량 output adaptation을 넘어 유용한 조건이 존재하는지, 그 local 이득과 현재 기여가 chronological future의 adaptation·update utility로 얼마나 전달되는지를 규명한다.”** 현재는 연구 질문이며 fresh 결과를 얻은 완료형 claim이 아니다.

## 데이터 노출과 아직 비어 있는 증거

[확인] [노출 장부](data_exposure_ledger.csv)는 실제 연구 기록·manifest의 source/target/기간/role을 추적하고 이미 결과를 본 구간을 final 후보에서 제외한다. 시작 시 있던 부정확한 장부는 [원본 사본](data_exposure_ledger.original_20260911_181304.csv)으로 보존했다. 특히 Study31은 BMRA가 아니라 BDG2와 Jena다.

[확인] 최종 장부는27행이며 [디렉터리 coverage](data_exposure_dir_inventory.csv)는 experiments28개/results27개, 총55개 PEFT 디렉터리를 연결한다. ALFRED·USCRN·초기 BDG2도 포함했다. Study12의 정확한 기간은 실제 prepared manifest에서 복원했다. [미검증] Full FT v1/v2는 version별 결과 디렉터리가 없어 v3/P0 계보에 보수적으로 연결했지만 개별 실행 노출을 완전히 재현하지 못했다. `results_viewed=yes`는 연구 과정의 결과 보고·분석 노출을 뜻하며 개인별 열람 로그를 뜻하지 않는다. 장부 행 수도 독립 실험 단위 수가 아니다.

[판정] [가용성 감사](holdout_availability.md)의 certified fresh unit은 현재0/2, 별도 sealed final unit도0/2다. Household 후속 기간 등 metadata 후보와 인증된 clean manifest를 구분한다. 이 결과는 외부에 데이터가 없다는 결론이 아니므로 `NO_CLEAN_HOLDOUT_AVAILABLE`을 선언하지 않았다. pretraining contamination은 UNKNOWN이다.

[미검증] 남는 증거는 (1) Time-PEFT 본문을 포함한 정확한 novelty 경계, (2) target-blind로 고정한 fresh unit 둘 및 별도 final reserve, (3) F0/HEAD/WIDE/JOINT의 fresh chronological 재현, (4) 동일-prefix C/U 불일치의 독립 재현, (5) 조건부 fresh-head/rolling 분석, (6) 가능하면 다른 backbone과 최종 미노출 family 검증이다. 기존 결과가 이 빈칸을 대신하지 않는다.

## Stage별 상태와 다음 실험 하나

- [확인] Stage0 저장소 감사 및 Phase0 목적·claim·story map 작성 완료. [계약](../../_docs/notes/tsfm_topics/07_research_direction/37_peft_paper_claim_gate_20260911.md).
- [확인] 기존 핵심 값 재구성 PASS. [판정] 노출 정보의 일부 historical timestamp와 외부 후보 인증은 미해결 상태를 장부에 남긴다.
- [판정] Novelty UNKNOWN이므로 Stage0~5의 전체 통과가 성립하지 않는다.
- [확인] [Stage A MD](preregistered_stage_a.md)/[JSON](preregistered_stage_a.json)은 **DRAFT_NOT_FROZEN / executable=false**다. 실제 기간·target·origin 간격·time-block interval·input/model hash가 비어 있으며 실행 가능한 preregistration으로 부르지 않는다.
- [확인] Stage A/B/C의 GPU·raw metric·paired contrast·uncertainty·실측 fit 비용은 미실행/미측정이다. 결과를 만들거나 G1/G2/G3 실패로 기록하지 않았다.
- [판정] **다음 실험은 정확히 하나: 상류 감사와 metadata 봉인 통과 뒤 두 fresh source-period의 F0/HEAD/WIDE/JOINT Stage A replication.** 그 전의 원문 확보와 clean manifest 감사는 실험이 아니라 실행 자격 확인이다. 현재 새 실험을 즉시 시작할 근거는 없다.

[판정] Stage A G1이 실패하면 adapter/selector 개발 STOP. G1 통과 후에만 Stage B, 현상 재현 후에만 Stage C, oracle/simple-rule/cost gate 모두 통과 후에만 method branch로 간다. 계획한 fit 수를 채우려고 후속 단계로 넘어가지 않는다.

## 실행 비용·파일·보존

[확인] 이번 **GPU fit0 / GPU forecast0 / GPU 실행 wall-clock0초**다. GPU 학습 최대 메모리는 **N/A(미사용·미측정)**이며 desktop 전체 GPU 사용량과 혼동하지 않는다. 최초 snapshot의 RTX4070 전체 메모리는673MiB/12,282MiB였으며 그것은 이 연구의 GPU 할당량이 아니다. CPU 핵심 재계산 wall-clock은 위3.910906초이며 조사·문서 작성 전체 시간과 구분한다. 작업 감사 구간은 [최종 저장소 receipt](repository_audit_final.json)에 기록한다.

[확인] 주요 새 산출물은 목적·claim37문서, novelty37문서, evidence JSON/MD/재현 script, 노출 CSV/감사/가용성 JSON·MD/재현 script, 미봉인 Stage A MD/JSON, story-flow audit, 이 readiness와 JSON, repository/hash/verification receipts다. [최종 검증](verification.json)은 파일·링크·hash·schema·git diff를 확인한다.

[확인] 기존 추적 파일의 변경은 연구방향 README 링크 추가와 `_docs/history/2026-09-11.md` 기록 추가뿐이다. 기존 실험 source/results/checkpoints/sealed artifacts는 수정하지 않았다. 원래 미추적 pytest 임시폴더와 tmp도 보존했다. commit/push, OS/보안설정 변경, 다른 프로젝트 프로세스 종료는 하지 않았다.

[판정] 중단/보류 항목은 Stage A(novelty·clean manifest 미충족), Stage B/C(선행 실험 미실행), method(반복성·예측성·강한 baseline 우월성 미확보), final(규칙 freeze·reserve 미확보), Study36 현재 DGP(기존 구성타당도 실패)다. Point/Hurdle 연구축은 이 논문으로 합치지 않았다.
