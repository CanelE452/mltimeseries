# Study20a: 보호 중단을 보존하고 저장 전 CUDA cache 정리

2026-09-09 04:08 KST. [20번 사전 계획](20_peft_fullft_reference_plan_20260909.md)의 데이터·방법·54fits·선택·지표를 유지한다. `peft_fullft_reference_v1` 최초 FULL_FT/Bike S0가 5updates 후 weights 저장 중 shared guard에 의해 중단됐다. 아직 production fit, C/E forecast는 0개다.

## 관측과 제한

시작19:01:36.338UTC, 중단19:01:46.583UTC. guard reason=`available_commit_below_limit`. 가용 RAM10.352GiB, GPU4049MiB/41°C, 가용 Windows commit4.942GiB였다. 시작 commit은10.365GiB다. 모델은 step0 V와step5 V를 출력했지만 `best_trainable.pt.tmp`437,379,072bytes만 남고 checkpoint 복원/완료 감사는 하지 못했다. 이를 완료된 fit으로 세지 않는다.

코드상 optimizer·gradient tensor를 해제하고 gc.collect해도 CUDA allocator가 빈 block을 reserved memory로 유지할 수 있다. FULL의 FP32 gradient와 두 Adam moment는 합약1.335GiB다. 이 cache와 Windows commit의 관계는 현재 로그로 확인되지 않은 가설이다. 2초 guard 검사 사이의 학습·최종V·저장 구간이 짧아 저장이 commit 하락을 일으켰다고 단정할 수 없다.

## 수치 변경 없는 복구

v1 모든 코드·계약·로그·부분 파일을 그대로 보존한다. 별도 `experiments/peft_fullft_reference_v2`와 `runs/peft_fullft_reference_v2`를 사용한다. 입력은 v1에서 확정한 동일 NPZ다. prepared manifest bytes도 동일하며 새 자료 준비·기간 선택은 없다. 기존 원천별 train90/V30/C20/E80과 native loss/optimizer/LR/seed/batch/200steps를 모두 유지한다.

v2 train의 실질 변경은 optimizer 삭제와 gc.collect 다음, checkpoint 저장 전에 `torch.cuda.synchronize(); torch.cuda.empty_cache()`를 실행하는 것이다. 살아 있는 모델 가중치나 CPU best state를 삭제하지 않는다. 학습 마지막 tensor 참조를 정리하고, load·snapshot·첫/마지막 update·cleanup전후·reload후에서 CUDA allocated/reserved, 시스템 가용 commit/RAM, process RSS/private commit을 기록한다. 이는 cache 반환의 실제 효과와 활성 학습 중 메모리를 구분하기 위한 계측이다. 모든 arm에 같은 정리를 적용하고 시간 비용에 포함한다.

기존 guard의 RAM5GiB/commit6GiB/childRSS8GiB/GPU10500MiB·85°C 기준을 그대로 유지한다. 페이지파일·드라이버·Defender·무관 프로세스 설정 변경은 없다. S0 six fits를 새 namespace에서 처음부터 검사한다. 메모리 기준을 넘으면 그 시도를 보존하고 중단하며, production을 강행하지 않는다. 원 최초 실패10.25초는 총 시도 비용에 포함한다.

원20의 가설·성공 해석은 동일하다. 복구 자체가 GPU 메모리 최적화 연구 결과나 크래시 원인 규명을 뜻하지 않는다.
