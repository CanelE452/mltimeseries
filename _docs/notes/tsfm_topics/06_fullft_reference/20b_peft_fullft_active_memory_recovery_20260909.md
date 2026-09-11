# Study20b: 학습 중 CPU best-state 보관과 CUDA cache 메모리 축소

2026-09-09. [20계획](20_peft_fullft_reference_plan_20260909.md)과 [20a 복구](20a_peft_fullft_memory_recovery_20260909.md)의 자료·학습 수치·선택·통계 설계를 유지한다. Production/C/E forecast는 아직0개다.

## 새로 확인한 사실

v2 Bike FULL_FT S0는 5updates와 저장·복원까지 종료했고 guard도 완료로 기록했다. 그러나 상세 계측은 active training에서 commit 여유가 step1 5.375GiB, step5 4.976GiB임을 보여 주었다. 2초 guard 주기가 짧은 학습 구간을 놓친 것이므로 이 완료 상태를 장시간 학습의 안전 근거로 쓰지 않는다. 정리 전4.967GiB→정리 후6.674GiB로 cache 반환 효과는 실제 확인됐다. 근본적으로 학습 중 메모리도 줄여야 한다.

Root는 다음 LoRA S0 중 runner를 중단했고 v2 runner32352/child38716의 동일 프로세스가 더 이상 존재하지 않음을 확인했다. 이전 guard/status는 덮어쓰지 않고 `interruption_receipt.json`에 이유와 stale lock 검증을 기록했다. v1/v2의 모든 소스·계약·부분 결과·성공/중단 기록은 보존한다. v2의2개 guard 완료만으로 전체 S0 통과를 선언하지 않는다.

## v3의 메모리 수명 변경

새 `peft_fullft_reference_v3`를 사용하며 기존 v1 prepared NPZ와 동일 manifest를 그대로 참조한다. CPU heap에 전체 best weights를 clone해 두는 대신, 파일 기반 NumPy memmap의 FP32 storage를 만들고 torch tensor view를 통해 GPU parameter를 직접 복사한다. 같은 working file의 내용을 V가 개선될 때만 갱신한다. 마지막 best_trainable.pt는 한 번만 저장하며 reload는 `mmap=True`로 읽는다. Forecast reload도 같은 방식이다. Working file은 실행 증거로 남기고 checkpoint 파일 크기와 별도로 보고한다.

매 optimizer update 이후 gradient를 `set_to_none=True`로 해제하고 CUDA cache를 반환한다. 다음 update의 모델·AdamW 상태·sample 순서는 유지된다. 동일 처리를 모든 arm에 적용하고 그 시간은 실제 step 비용에 포함한다. CPU best-state의 저장 매체와 사용이 끝난 GPU memory의 반환 시점을 바꾸는 것이며, FP32 가중치/BF16연산·native objective·AdamW hyperparameter·micro4+4·effective8·LR/seed/200updates를 바꾸지 않는다.

전체 모델의 가장 작은 LR S0에서 원 v2와 step0/5 V 수치·예측·sampler·복원된 trainable tensor hash를 비교한다. S0는 train-only이며 이 비교에 C/E를 사용하지 않는다. 그 외 원천/arm S0도 원 계획의 실제 update·scope·finite gradient·복원 감사를 그대로 통과해야 한다.

## 더 촘촘한 자원 확인

기존2초 shared guard와6GiB commit 하한을 유지하면서 각 microbatch backward 직후 가용 commit을 빠르게 확인한다. 하한 미만이면 child가 즉시 중단한다. Loaded/snapshot/update/cleanup/reload phase를 함께 기록한다. mmap의 Windows private commit 절감량과 활성 학습 메모리는 재실행 전에는 가설이며 S0 실측으로 확인한다. 이 검사를 통과하지 못하면54fits를 시작하지 않는다.

페이지파일·프로세스 강제 정리·드라이버·Defender 변경이나 한도 완화는 하지 않는다. 이전 중단은 리소스 고갈 크래시로 바꾸어 서술하지 않는다. 최초보호중단10.25초, v2 완료 시험 및 별도 중단시도, v3 전체비용을 구분한다. 종료시각이 없는 중단시도는 정확한 elapsed를 지어내지 않는다.
