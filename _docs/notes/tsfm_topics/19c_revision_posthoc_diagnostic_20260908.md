# 19c. CPU 평가 후 정규화 차이 분리 진단

2026-09-08. 원19 CPU 실행이 23:02:44~23:02:51 KST에 정상 종료했다. 사전 기준인 correction-vs-FIRST 평균 개선 1% screen은 통과했지만 ZERO_GROWTH가 두 계열에서 모든 ridge보다 낮은 E MSE였다. PAYEMS의 FIRST는 V에서 λ=1, REVISED는 λ=10을 선택했다. 따라서 15.887% 정규화 평균 개선을 label revision 자체 효과로 해석할 수 없다.

이 문서는 **E 결과를 본 뒤 작성한 사후 진단**이다. 원19 계획·선택·점수·판정을 바꾸지 않는다. GPU는 아직 실행하지 않는다.

- 각 계열에서 FIRST_FIXED_X와 REVISED_FIXED_X가 V로 선택했던 두 config의 합집합만 사용한다. PAYEMS는 λ1/10·expanding, INDPRO는 λ100·expanding이다. 새로운 E sweep이나 새 λ 선택은 하지 않는다.
- 각 고정 config에 대해 first label, current capped label, first label + 당시 성숙 사례 평균 revision 보정을 비교한다. 세 arm의 feature/row membership/window/λ는 동일하다.
- 2020~2024 전체 60 origin을 유지한다. 계열별 MSE/MAE, label-only 같은 config 차이, 실제 예측 최대차이를 보고한다. 연도별 손실 분해는 사후 설명이며 COVID 기간 제외 점수로 주결과를 교체하지 않는다.
- 목적은 원19 개선의 정규화 선택·평균 보정·label revision 기여를 구분하는 것이다. 유리한 새 결과를 발견하거나 새 PEFT 필요성을 주장하는 추가 평가가 아니다.

새 소스 및 출력은 별도로 저장하고 원 CPU 계약의 9개 소스는 수정하지 않는다.
