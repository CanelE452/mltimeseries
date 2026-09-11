# mltimeseries

**완료: [36번 순서 의존성 PEFT Phase 0 게이트](notes/tsfm_topics/07_research_direction/36_dependence_phase0_results_20260911.md).** 현재 전/역 3상태 Markov DGP는 GPU 진입 전 중단. 주변분포·ACF/PSD·Bayes risk 통제는 성립하지만 부호 재라벨링과 정확히 동치이고, train-only 전이표가 oracle 대비 평균0.0032% excess risk에 그쳐 내부 PEFT 필요성을 검증하지 못함. GPU 0회, adapter/외부검증 미진입.

**완료: [35번 head 학습량·성분 개입 검증](notes/tsfm_topics/07_research_direction/35_head_convergence_20260911.md).** 180→720step 대조 완료. LoRA는 동일 용량 head보다 평균8.804%F0 좋지만 F0보다4.305% 나빠 G1 미통과. head가 따라잡은 결과는 아니며,12개 선택 모델 모두 train/V 개선과 D 악화. 조건부 head재학습·warm-start 미실행.75 GPU 실행·독립 검산 완료,학습 종료.

**완료: [34번 초기 LoRA 이득 진입 검사](notes/tsfm_topics/07_research_direction/34_initial_headroom_20260911.md).** 초기 LoRA 이득 G1은 4조합 중3조합 통과, controller 여지 G2는 +0.0408/+0.0756%F0로 미통과. 평균 동일 용량 head 대비 +2.428%F0, F0 대비 +1.418%F0. 일부 head의 학습 상한 선택으로 수렴·표현 원인은 미확정. 123 GPU 실행·독립 검산·5개 그림 완료, 학습 종료.

**완료: [33번 시기 전이·신호 제거·실제 비용 검증](notes/tsfm_topics/07_research_direction/33_decision_transfer_20260911.md).** 177 GPU 실행·독립 검산 완료, 세 기준 미통과. 후보는12/12 STOP과 같은 모델을 반환했고 조기 종료보다20.1~33.7% 느렸다. 동일 용량 head 대비 평균 LoRA 이득+.944%F0는 Jena/BMRA에서 부호가 다르다. 4개 그래프·전체 결과·실패 원인과 다음 연구 순서 정리; 이번 학습 종료.

**완료: [32번 미래 LoRA 업데이트 가치 진단](notes/tsfm_topics/07_research_direction/32_future_utility_20260911.md).** 12조건·24분기와 독립 검산 완료. 현재 기여가 양수여도 미래 업데이트가 불리한 사례16개, checkpoint 선택 후21/24 같은 결과. 새 방법의 우월성은 미확보; 결과·그래프·논문 요건 정리.

시계열 ML 연구 프로젝트. `covariate-trust-pilot` 과 저장소·기록을 분리한다.

시작: 2026-09-06

## 현재 상태

**2026-09-11 완료: [31번 단일 LoRA 적응량 검증](notes/tsfm_topics/07_research_direction/31_contribution_freeze_20260911.md).** 48fit+50E+S0+13시간재실행,112guard exit0. 기여 기반 동결 비용기준실패(7.42/3.77%절감);고정동결19.92/21.46%절감·평균성능우세. 원13학습과재실행전체곡선/예측완전일치,비용보완·독립검산·그래프완료. 후보E예측11/12는FULL동일. 현재규칙채택안함·다음은같은상태에서미래업데이트가치분리진단. 아래는 이전 이력이다.

**2026-09-10: 동일 파라미터 head·짧은 적응 반응 완료.** [30번](notes/tsfm_topics/07_research_direction/30_capacity_probe_20260910.md): WIDE36학습+shortALL24학습+WIDEE24평가+smoke1=85guard 정상종료. P1 Bike FULL90 추가LoRA +4.358/+5.200%F0, 평균4.779;같은학습파라미터수대조 gate 통과. P1 probe방향11/12<항상ALL12/12,6cell rank0.543;품질기준통과/시간6.397~14.294%증가로비용기준실패. 두E노출개발자료,새방법아님. 독립61fit/48probe/52예측검산,3그림,근거23파일 완료. 진행중학습0/guard lock없음. 아래는 이전 이력이다.

**2026-09-10: 1→2→3 완료.** [28번](notes/tsfm_topics/07_research_direction/28_optimization_control_20260910.md)의48학습/48평가 최적화 gate 통과 후, [29번](notes/tsfm_topics/07_research_direction/29_overlap_transfer_20260910.md)의24학습/26예측 새 E 검증을 완료했다. OVERLAP은11.54/12.63% 비용 절감, regret0.503/0.674%F0로 정확도 gate 실패. 같은 head 구조 대비 LoRA 추가 이득12/12양수 및 SPREAD>RECENT 순위 반전 확인. 두 단계149guard exit0, 독립 검산·그림·기록 완료, 해당 학습프로세스0/guard lock없음. 새 adapter나 인과 일반화는 미확보. 아래는 이전 이력이다.

**최신 2026-09-10: R1 구간 구성·과거 잔차 검증 완료.** [27번 결과](notes/tsfm_topics/07_research_direction/27_r1_coverage_residual_validation_20260910.md): 새16 fits/16 평가와 기존8 모델 재사용. 추가LoRA 이득 FULL90/SPREAD30/RECENT30은 Bike +4.171/−1.019/+1.267%F0, Household +.418/−.139/+.167%F0. 작은자료16 fits에서9개step0/7개step40선택으로 최적화 설명 남음. Bike bias보정 E+.497%,HouseholdZERO. 새adapter/독립test미확보. CPU float32 간격검사 오류1회는 별도float64 사본으로복구,재학습0. 독립26점수 최대차5.55e-17. 진행중학습없음.

**최신 2026-09-10: R1–R3 통제 실행 완료.** [실제 결과](notes/tsfm_topics/07_research_direction/26_mechanism_diagnostics_20260910.md): 36 fits/24 평가, 같은 MLP 대비 LoRA 추가 이득 Bike 4.171%F0/Household .418%F0. V로 고른 EARLY는 F0 대비 4.404% 개선, 동일 예산 ALL_LOW는 4.528%여서 위치 선택 우위 미확보. Hospital 월 제외에서는 약 75%가 안정적이지만 그 집단에서도 미래 개인화 평균 손해. 원인과 새 방법은 미확정. Runner 32분46.6초, 63 guard jobs exit0, 안전 중단0, 독립 점수 오차0. 진행 중인 학습/commit/push 없음. 아래 Hospital 문단은 앞선 완료 이력이다.

**2026-09-10: Hospital 재개·4 fits·E1/E2·독립 감사 완료.** [최신 결과](notes/tsfm_topics/08_hospital_shared_strength/24_hospital_results_20260910.md): 표준 shared LoRA는 F0 대비1.183% 개선했지만 INDIVIDUAL은 GLOBAL보다0.412%F0 나빴다. 두 seed·두 평가 연도에서 주효과 모두 음수. 현재 직접 개별 강도 선택의 추가 효용은 미확보이며 새 방법을 주장하지 않는다. 사용자 승인 앱 정리와 작업별 commit13 GiB 진입 후 새8 stages 안전 중단0, 독립48점수 재계산 PASS. 추가 학습은 없다.

[전체 과정의 재개 전 스냅샷](notes/tsfm_topics/research_review_20260910/README.md), [주제별 목록](notes/tsfm_topics/README.md)에 기존 연구를 보존했다. 아래는 당시 판단을 남긴 이력이다.

문헌 사전조사 자료 두 건(아래 "자료")에 더해, 2026-09-06에 `_docs/notes/tsfm_topics/`
아래 연구 주제 노트 3건이 들어왔고 그중 1번(예측 목표 조건부 토큰화)의 1차 파일럿을
실행했다. 계열별 최신 상태는 아래 "계열 현황"에서 본다. 최초 주제 노트의 frontmatter는 이후 실행 상태를 반영하지 못한 경우가 있으므로 실제 결과와 history를 먼저 확인한다.

2026-09-07에는 PEFT 후보 A/B/C를 같은 notes 폴더의 04~06번으로 보존하고 A의 실험 설계 초안을 추가했다. 09-08에는 A S1 62개 실행을 완료했으며 판정은 INCONCLUSIVE다. 이후 출력부 보완 투영 후보는 선행연구·CPU 진입 검사에서 새 방법 주장을 중단했다. 통제 합성 실험124개와 Q00 native LoRA 삭제 대조12개, train-only 지연 추정 후속18개도 완료했다. Q00에서 attention LoRA만으로 결합 LoRA의 성능을 유지했고, 같은 분포의 새 표본에서는 추정 지연 선형 회귀가 oracle에 실용적으로 근접했다. 이 Gaussian 조건의 새 adapter 탐색은 중단하며 time/group 위치 가설이나 외부 데이터에서의 신규 PEFT 필요성은 입증하지 못했다. 아래 최신 결과 링크에서 근거를 본다.

### 2026-09-08 반복 탐색 시작·C 단순 보정 진단 완료

사용자가[후보별실험→실패분석→자기평가→다음실행](notes/tsfm_topics/07_research_direction/11_peft_topic_search_protocol_20260908.md)의지속탐색을승인했다. [C폐쇄진단](notes/tsfm_topics/03_selection_calibration/11_peft_calibration_closure_results_20260908.md)은새GPU학습없이24개보정결과를검증했다. ALIGN_ATTN coverage69.97→79.82%,score.368454→.363769로단순21분위수offset보정기준을통과했다. 이미본Gaussian평가/선택validation재사용이므로외부확증이아니며,이관측만을위한새PEFT필요성은약하다.

이어[외부원천screen12](notes/tsfm_topics/02_adaptation_scope/12_peft_external_gap_results_20260908.md)의28fit·14forecast·RAW2개를완료했다. 보정후H대LoRA효과는Bike+3.9253%F0/97.5%CI[2.0523,5.1823]%,Household+1.3553%/CI[−.0682,3.2396]%다. RAW는둘다열세지만전체두원천진입기준은미충족이다. QCAL은모든선택FM의proper score를악화시켰으므로불확실성해결로해석하지않는다. 본GPU wall13분4.7초,최대GPU2619MiB/58°C,대상Windows이벤트0건이었다. CPU센서null처리와그림상수로그축문제는산출물을보존하며수정했고재학습없었다.

초기 후보 AirQuality는 기존 fev 평가 노출을 확인해 새 원천 대조에서 제외했다. [시간 블록 반복 13의 결과](notes/tsfm_topics/02_adaptation_scope/13_peft_temporal_replication_results_20260908.md)도 완료했다. H 대 LoRA 효과는 Bike+7.416%F0/97.5%CI[4.550,11.263]%, Household+1.776%/[0.531,3.147]%로 전체 실용 진입 기준은 미충족이다. Bike LoRA는 F0보다 좋지 않았고 큰 H 대 LoRA 차이는 H의 악화에서 나왔다. 사전 규칙에 따라 현재 A 분기를 닫는다. 이는 PEFT 전체의 반증이 아니다.

13번은 28 fit·14 forecast·RAW2·CPU29검사와 독립 수치 검산을 완료했다. 마지막 추론에서 Git56개>한도32로 안전 중단1회 후 같은 가중치로 그 추론만 재개했다. 전체43GPU시도 guard942.205초, 점검 대기 포함 경과18분13초이며 실패 로그를 보존했다. 최대GPU2732MiB/58°C, S0 이후 조회 대상 Windows 이벤트0건이다. 성공 작업만 담는 자원 파일과 전체 시도를 담는 recovery_audit를 구분해야 한다.

[B 최소 진단14 결과](notes/tsfm_topics/03_selection_calibration/14_peft_selection_regret_results_20260908.md): 추가학습 없이 미선택28추론을 457.341초/모두exit0으로 완료했다. 저장40후보/80절차 점수와 네 주효과 CI는 독립 NumPy 재계산과 오차0이었다. Bike13 선택 손실은+3.079%F0/98.75%CI[0.414,5.838]%지만 사전1% 하한과 두 원천 반복을 통과하지 못했고 고정LR1e-5의 단순 규칙 veto가 성립했다. 현재 B 분기를 닫는다. 이는 사후 전체후보 oracle gap이나 PEFT 연구 전체의 부재를 뜻하지 않는다. CPU47검사·4subtests/S0차이0/원557hash보존, 안전중단0·대상Windows이벤트0을 확인했다. [다음 후보의 선행 경계](notes/tsfm_topics/07_research_direction/14_next_method_literature_boundary_20260908.md)를 검토 중이며 전체 주제 탐색은 미완료다.

### 2026-09-08 목적함수 정렬 진단15 완료

[15번 결과](notes/tsfm_topics/04_objective_observation/15_peft_objective_alignment_results_20260908.md): 같은 OFF_LORA·LR3e-5·200updates에서 NATIVE/NORM_ALIGNED/RAW_ALIGNED를 비교한 6fit·6forecast와 S0 6개를 완료했다. Raw 학습의 추가 효과는 Bike−0.218%F0/97.5%CI[−0.680,+0.142]%, Household+0.122%/[−0.059,+0.255]%로 사전 두 원천 +1% 하한 기준을 통과하지 못했다. 현재 objective-alignment screen을 닫으며 raw-loss 자체를 새 방법으로 주장하지 않는다. LoRA의 F0 대비 이득과 새 손실의 추가 이득은 구분한다.

[확인] CPU53검사, 원12 native S0/본 trajectory exact replay, 보호885hash, 분석1,057입력hash, 독립16점수 오차0/주 CI 오차2.62e-16을 확인했다. 최종 검증을 포함한 본 runner458.546초/CPU분석36.218초, GPU 최대 관측2581MiB/57°C, 여유 RAM 최소13.924GiB였다. 18개 GPU시도 모두 exit0, 안전중단·재시도0회이며18:06:06~18:18:42 대상Windows이벤트0건이다. 다음 후보는 한 시퀀스 내 이종 관측 연산의 식별가능성과 단순 대조의 충분성부터 검토하며, 아직 새 방법이나 해당 학습 결과는 없다.

### 2026-09-08 관측 연산 진입 검사16 완료

[16번 결과](notes/tsfm_topics/04_objective_observation/16_observation_operator_entry_results_20260908.md): CPU 수학 감사6.078초/exit0 및 USCRN Tucson2024 두 파일 다운로드·QC10.094초/exit0을 완료했다. 새 학습0회다. 알려진 AR(1)에서 직접 Gaussian 조건부 계산과 순차 Kalman의 미래 gain/평균/공분산 차이는4.36e-15/3.89e-15/5.55e-16이었다. 전체CPU17검사와 별도52차원 precision146검사(최대오차3.07e-14)를 통과했다. `EXISTING_LINEAR_CONDITIONING_SUFFICIENT`는 이 알려진 조건의 진입 판정이며 새 PEFT의 성능 결과가 아니다.

실제 자료8,784시간/105,408개5분 격자의 중복·격자 누락은0이었다. 같은 마지막5분 평균은8,087쌍 전부 같았고, 완전한8,076시간의 mean12 대 시간 평균은 최대0.125°C로 단순 반올림 범위 밖 예외 한 건이 있었다. 결측·제품 revision·실제 공개 지연의 한계를 보존한다. 사전 첫 hour 경계 예상과 달리 파일은01:00/00:05부터 다음 해00:00까지여서 이전 연도 파일은 필요하지 않았다. 실패·재시도·안전중단0,18:48:07~18:58:42 대상Windows이벤트0건이다. 전체 주제 탐색은 미완료이며 다음은 실제 coarse-only 대상의 정보 계약과 단순 profile/head 대조의 필요성 검토다.

### 2026-09-08 월합 감독 실험17 완료

[17번 결과](notes/tsfm_topics/04_objective_observation/17_coarse_supervision_results_20260908.md): BDG2 두 site의 16 target에서 월합만으로 head를 적합하고 고정 head에 attention LoRA를 추가했다. Train176/V48/E48창, E34,944시간이 모두 유효했다. Head lambda.001/수준보정lambda10, V 단순 policy COARSE_LIFT, LoRA step40을 사전 규칙으로 선택했다. 본200 update·평가5arm을 완료했으며 현재 조건의 판정은 CLOSE_CURRENT_COARSE_SUPERVISION_SCREEN이다.

[확인] Eagle MSE는 F0.05749→head1.59680→LoRA1.59697, Lamb는.23261→.51539→.51472다. 큰 악화는 head에 이미 존재했다. Eagle의 head 월평균 오차는 조금 줄었지만 패턴 오차는 약29.78배였다. 독립 검산은 원 E값 오차0, 960cell 지표 차이4.44e-16, CI차이8.88e-16을 확인했다. Ridge의 12,304 명목계수 대비 실제 월합 design rank24, train/V에도 큰 평균제거 보정이 존재함을 확인했다. 8슬롯 대비만을 단독 원인으로 볼 근거는 없으며 대부분은 patch간 공통 성분 변화다.

CPU68검사/17.24초, cache16.234초, S0 14.234초, 본학습78.828초, 캐시 이후 전체 runner126.624초였다. 다운로드~그림9guard 합164.030초/모두exit0, 관측GPU2330MiB/55°C·여유RAM최소14.044GiB, 조회대상Windows이벤트0건이다. 새 방법은 미확보이고 전체goal은 active다. 다음 판단은 같은 월평균을 유지하는 단순 패턴 보존 대조가 충분한지이며, 현재 E에 새 LR/anchor sweep을 추가하지 않는다.

### 2026-09-08 Train-only 지연 추정 후속 완료

[결과와 연구 방향](notes/tsfm_topics/02_adaptation_scope/10_peft_trainlag_results_20260908.md): 새18GPU실행(12fit+6F0/cache), CPU RAW3개, S0네 개와 CPU31검사를 완료했다. 같은1024 train 미래label로113개 지연을 검색했고 세 corpus 모두Y/U/V32/48/48을 선택했다. 새 난수 표본은 같은 Gaussian family이며 외부 원천은 아니다. 평균 score F0 .536967, ATTN .497214, ALIGN_F0 .460572, ALIGN_ATTN .368454, RAW .323352, ORACLE .322664다.

정렬 대 원 attention 효과는+6.8240%F0(97.5% CI+5.0581~+8.5596%), 정렬 후 LoRA 추가효과는+17.1551%(+15.7412~+18.5683%)였다. 별도 RAW−oracle gap은0.1282%F0(95% CI0.0690~0.1863%)로 사전1%근접 문턱을 통과했다. 이 조건에서 새 PEFT 필요성이 약하므로 adapter 탐색을 중단한다. ALIGN_ATTN80%coverage69.97%의 undercoverage는 남는다. 본학습11분40초/모든guard exit0, 자원 표본 GPU최대2295MiB/53°C, RAM여유최소14.99GiB, S0이후 대상 Windows이벤트0건이었다. 실제 유보 원천에서 단순 풀이를 넘어서는 문제 확인은 다음 진입 기준이며 이번에는 미실행이다.

### 2026-09-08 Q00 native LoRA 모듈 삭제 대조 완료

[결과와 연구 방향](notes/tsfm_topics/02_adaptation_scope/09_peft_module_ablation_results_20260908.md): 새 학습12/12, S0세 개 및 수치 검증 완료. 고정 LR3e-5에서 attention 삭제 비용은 F0의+6.4615%(97.5% CI+5.3183~+7.5690%), output projection 삭제는−0.0416%(CI−0.0517~−0.0309%)로 조건부 실용 동등 범위±1% 안이었다. ATTN_ONLY score0.496517, BOTH0.496737, OUT_ONLY0.531052, F0 0.531068이다.

Native 출력 LoRA를 제외해도 이번 성능이 유지되지만 파라미터 감소는2.26%라 새 효율적 방법의 기여로 주장하지 않는다. 과거 공변량 회수 격차는 남으며, 다음 방법론 진입 전에 train-only lag search 등 단순 풀이가 이를 해결하는지 새 원천에서 판단할 것을 권한다. 본학습7분33초, GPU 표본 최대2262MiB/55°C, 여유 RAM 최소15.31GiB였다. 최초 분석의 문자열 ID 처리 오류는 로그를 보존한 채 분석기만 수정했고 최종 검증은exit0다. 원 학습 결과와 소스 계약을 유지했다.

### 2026-09-09 방법 파일럿 인계 Part A — 후보 스크린 `NO_METHOD_SPECIFICATION`

외부 지시문(`PEFT_METHOD_PILOT_CLI.txt`)의 Part A를 수행했다. [로컬 인벤토리 21](notes/tsfm_topics/07_research_direction/21_peft_method_pilot_local_inventory_20260909.md)(지시문이 전제한 kit 폴더 부재 포함), [후보 스크린 22](notes/tsfm_topics/07_research_direction/22_peft_method_candidate_screen_20260909.md)(네 방향 A/B/C/D 전부 닫힘, 셋은 새 GPU 학습 없이 저장 결과만으로; 판별자 검토 후 방향 B 근거 정정), [다음 경로와 의견 23](notes/tsfm_topics/07_research_direction/23_peft_next_paths_and_opinion_20260909.md)(문제 명세 + 짧은 계열 풀 예측, 백본 검사 3 fits → break-even 4 fits, 조건부 후보 P1~P3, 승인 항목 A~E). 새 GPU 학습 0회, 기존 study 수정 0건. Part B는 승인 전 미진행.

## 계열 현황

- **01_forecast_query_tokenization:** 27/27 fit 완료, `INCONCLUSIVE`. 주요 대조 macro −0.037% [−0.128%, +0.033%], 사전 문턱+1.0%, 8조건 중3개 통과(09-06). [초기 노트](notes/tsfm_topics/01_initial_topics/01_forecast_query_tokenization.md).
- **02_observation_aware_resolution:** 작은 target-trained 모델의 interval-integrated Fourier 표현 pilot 및 감사 완료. `INCONCLUSIVE` / `STOP_SCALING_CURRENT_INTERVAL_INTEGRATED_FOURIER_O` / 넓은 질문 `OPEN_NOT_DIRECTLY_TESTED`. 본 결과는 현재 작업트리에 없고 commit `4c6c805`의 OA 결과·audit에 보존돼 있다. FM PEFT·한 시퀀스 안의 이종 관측 연산·관측 likelihood는 미시험이다. [실행 이력](history/2026-09-06.md), [초기 노트](notes/tsfm_topics/01_initial_topics/02_observation_aware_resolution.md). Dry-run 판정을 본 결과와 혼동하지 않는다.
- **03_uncertain_future_covariates:** ECMWF/ENS GB2019–21와 BMRA8농장, 7방법×3seed=21fit 완료(09-07). `NO_INCREMENTAL_PATH_VALUE`: P 대 D−1.194%/95%CI[−2.252,−0.270], P 대 S−2.046%/[−3.726,−0.555]. 평균 예보 M이 가장 좋았다. [실제 결과](../results/uncertain_covariate_path_pilot_v1/STATUS.md), [판정](../results/uncertain_covariate_path_pilot_v1/verdict.json), [실행 이력](history/2026-09-07.md). 모든 방법은 같은 target-history FM cache를 사용했고 미래 날씨는 외부 adapter가 처리했다. 약50배 차이는 adapter-only 지연이며 K회 backbone 압축 실험이 아니다. 실제 발행·계량 공개 지연을 완전히 검증한 배포 결과도 아니다.

09-08 후속 후보 검토에서 기존 02/03의 완료 기록과 이 인덱스가 어긋난 것을 확인해 정정했다. 이미 평가한 원천·기간을 다시 미사용 holdout으로 취급하거나 같은 표현을 새 후보처럼 재실행하지 않는다.

## 자료

| 파일 | 내용 |
| --- | --- |
| [`reference/time_series_research_dossier_20260906.md`](reference/time_series_research_dossier_20260906.md) | 2025년 말–2026년 시계열 논문·파운데이션 모델 사전조사. 업로드 포스터 4개 재독해, 48개 외부 항목, 5개 후보 연구군, 비교 시 필수 구분 |
| [`reference/paper_catalog.json`](reference/paper_catalog.json) | 위 조사의 48개 항목을 기계가 읽을 수 있게 정리 (`id` / `venue_status` / `mechanism` / `inspection` / `interpretation_limit` / `primary_sources`) |

두 파일은 업로드 원본을 그대로 옮긴 것이다 (배치 시 sha256 일치 확인). 이후 dossier 의
포스터 섹션 헤더 4개에서 발표자 실명·원본 PDF 파일명·학회 포스터 번호를 지우고
등장 순서대로 `포스터 1`~`포스터 4` 로 바꿨다 (2026-09-06, 개인정보 제거).
그 4줄 외에는 내용을 고치지 않는다.

## 2026-09-07 추가 조사 — Foundation Model PEFT

- [PEFT 조사와 연구 방향](reference/time_series_peft_research_dossier_20260907.md): 시계열 적응 논문 17개, 일반 PEFT·보존 비교 8개, 공식 모델·구현 3개, 본문 미확인 공개 artifact 1개를 구분한 총 29개 항목. 핵심 문헌의 비교 조건, 코드 경계, 반증 가능한 가설 A/B/C와 실행 순서를 정리했다.
- [PEFT 카탈로그](reference/time_series_peft_paper_catalog_20260907.json): 항목별 출처·확인 범위·해석 한계, 고정 코드 revision, 로컬 코드 hash 및 DOI 검증 결과.

현재 추천은 native multivariate FM에서 **출력 보정으로 충분한 변화와 temporal/group 내부 갱신이 필요한 변화가 구분되는지 먼저 확인하는 것**이다. Novelty와 실제 이득은 미검증이다. 선택 안정성(B)과 예측 능력 보존(C)은 관찰된 실패에 따라 좁힐 후속 후보이며, 이번 조사는 학습·환경 설치를 실행하지 않았다.

이 추가 문서는 위 업로드 원본 2개를 수정하지 않고 별도로 작성했다. 아래 기존 자료의 해석 원칙은 계속 적용한다.

### PEFT 후보 노트와 A 실험 설계

- [A — 필요한 적응 위치](notes/tsfm_topics/02_adaptation_scope/04_peft_adaptation_scope.md): 우선 설계할 가설. [A 실험 설계 v0.1](notes/tsfm_topics/02_adaptation_scope/04_peft_adaptation_scope_experiment_plan.md)에 실제 개발 데이터, 출력/probe 대조, time/group 동일 예산, 합성 oracle, 시간·학습·판정 계약을 구체화했다.
- [B — 적응 범위 선택의 신뢰성](notes/tsfm_topics/03_selection_calibration/05_peft_selection_reliability.md): 14번 저장 후보 진단 완료, 현재 B 분기 종료.
- [C — 필요한 예측 능력의 보존](notes/tsfm_topics/03_selection_calibration/06_peft_forecasting_ability_preservation.md): 11번 기존 synthetic 보정 진단 완료, 해당 관측에서 새 PEFT 필요성이 약해 종료. 넓은 보존 문제는 미확증.

A의 추천 순서는 실데이터의 적응 여지 확인 → 통제 합성의 위치×변화 판별 → 새 원천·독립 backbone의 확증이다. 설계 문서의 숫자는 설계 제안이며 성능 결과가 아니다. 설계 당시에는 모델 학습과 환경 설치를 실행하지 않았다.

### 2026-09-08 PEFT A S1 실행 결과

[S1 결과와 후속 판단](notes/tsfm_topics/02_adaptation_scope/04_peft_adaptation_scope_s1_results_20260908.md)에 실제 결과를 정리했다. 본학습 62개가 68분 30초에 모두 정상 종료됐고, 최종 선택 결과 40개와 paired 효과 4개의 검증이 통과했다. 본학습 자원 기록의 GPU 온도 최대 70°C, 여유 RAM 최소 15.80 GiB, Git 프로세스 최대 2개였다.

A1 판정은 **INCONCLUSIVE**다. ETT LoRA의 선택 readout 대비 추가 이득은 F0 점수의 +2.14%, Jena는 -1.52%였지만 각각의 1·3·7일 block CI가 모두 0을 포함했다. FULL도 일관된 추가 우위를 보이지 않았다. S2의 대규모 위치 탐색에 앞서 더 긴 고정 평가 구간과 독립 원천에서 추가 효용을 확인하는 방향을 권한다. Cache 1-group/직접 평가 4-group의 작은 수치 차이는 별도 무학습 진단으로 재현했으며, 다음 확증에서 일치시킬 항목으로 기록했다.

### 2026-09-08 출력부 보완 PEFT 후보의 방법론 진입 검사

[선행연구·최소 구현 판정](notes/tsfm_topics/02_adaptation_scope/07_head_complement_method_gate_results_20260908.md): 출력부와 중복되는 LoRA 예측 변화를 제거한다는 후보는 SDDR/ONO/PHO와 핵심 연산이 겹친다. CPU 최소 구현의 8개 등가성 검사가 통과했고, 실제 ETT train 특징의 4-origin 배치 세 곳에서 rank 168/168 및 hard complement의 소멸을 확인했다. 전체 support는 소멸하지 않지만 별도 신규성이 남지 않아 GPU 학습으로 확장하지 않았다. 검증 guard 4.079초/exit0, stderr 없음. 기존 S1 source 계약과 결과를 보존했다. 이 결과는 해당 후보의 새 방법 주장을 중단한 것이며, TSFM 예측 성능의 열등성을 측정한 결과가 아니다.

### 2026-09-08 통제 PEFT 실패 조건 실험 완료

[통제 실험 결과와 방법론 판단](notes/tsfm_topics/02_adaptation_scope/08_peft_shift_mechanism_results_20260908.md): 네 조건×세 train corpus의124개 실행(112 fit+F0/cache12)을67분53초에 완료했다. 선택 결과132행의 입력·소스·선택·점수 재계산 검증이 통과했다. 본실행 자원 표본에서 여유 RAM 최소15.16GiB, GPU 온도 최대58°C, Git 최대2개였고 대상 Windows 크래시 이벤트는0건이었다.

일반 LoRA의 F0/선택 head 대비 개선은5.64~7.74%로 네98.75% CI 모두 운영 문턱1%를 넘었다. 새 head를 포함한 여섯 절차의 최종 선택72개는 전부step0/F0와 같았다. 그러므로 time/group 차이0은 모듈 역할 동등성의 증거가 아니다. OFF에는 native 출력층 LoRA도 포함돼 내부 attention의 필수성을 입증하지 못한다.

추가 학습 없는 과거 입력 정렬은 F0 대비12.17~15.50% 개선됐고, 알려진 lag 사전을 가진 RAW는 oracle 평균 MSE0.0024~0.0026까지 회수했다. “관측된 과거 공변량 정보의 활용”으로 문제를 좁히되, LIFT·ChronosX 등 기존 풀이와 단순 회귀를 넘어선 새 PEFT의 필요성은 아직 입증되지 않았다. 새로운 time/group routing 방법 제안은 보류한다. [원 계획](notes/tsfm_topics/02_adaptation_scope/08_peft_shift_mechanism_plan_20260908.md), [탐색 진단 추가 계획](notes/tsfm_topics/02_adaptation_scope/08_peft_shift_mechanism_diagnostic_addendum_20260908.md), [실행 README](../experiments/peft_shift_mechanism_v1/README.md).

## 이 자료를 어떻게 쓰고 어떻게 쓰지 않는가

dossier 자신이 경계를 명시하고 있고, 그대로 따른다.

- 대부분 항목은 **채택 상태와 공식 초록 수준의 스크리닝**이다. 전수 독해·재현·검산이 아니다.
- 성능 수치는 출처가 보고한 값이며 이 조사에서 실행해 확인한 값이 아니다.
- 메인 학회 채택 / workshop / arXiv 공개본 / 기업 공개를 구분한다.
  `arXiv 공개본; 정식 채택 미확인` 은 심사 중이라는 뜻이 아니다.
- 특정 검색어와 관심사에 따른 사전조사이지 체계적 문헌고찰이 아니므로,
  분야 전체의 빈도·비율을 추정하는 데 쓰지 않는다.
- **novelty 확정이나 새 실험 실행 승인의 근거로 쓰지 않는다.** 대표 논문을 깊게 볼 때는
  초록의 성능 주장을 옮기지 말고 target·input·split·checkpoint·compute·metric·
  baseline·ablation 을 추출한다.

## 아직 확정하지 않은 5개 후보 연구군

dossier 에 있는 그대로 옮긴다. 연구 기회가 남아 있다고 증명된 것이 아니라,
기능적으로 가장 가까운 선행연구를 더 읽기 위한 후보군이다.

| 연구군 | 대표 비교 문헌 | 다음 독해에서 답할 질문 |
| --- | --- | --- |
| Fine-tuning/adapter | SFF, Time-PEFT, CoRA, UniCA | LoRA/Full FT 보다 나아지는 원리가 무엇이며 최신 native multivariate 파운데이션 모델에도 남는 문제인가? |
| Recurrent/continuous forecasting | TiRex-2, FlowState, Toto 2.0 | state/context/해상도/horizon 을 어떤 구조로 처리하며 공개 코드가 실제 지원하는 범위는 어디까지인가? |
| Generative output/training loss | Sundial, MMPD, DBLoss, DistDF | 개선 대상이 point mean, quantile, joint trajectory 중 무엇이며 어떤 강한 baseline 을 넘어서는가? |
| Multimodal/related-context transfer | UniCA, VisionTS, TimeOmni-VL, In-Context Fine-Tuning | 추가 문맥 정보가 수치 예측을 실제로 개선하는가, 정보량·계산량 통제 후에도 효과가 남는가? |
| General representation | TSPulse, GTM, Zeus, CauKer | forecasting 외 분류/복원/이상탐지까지 범위를 넓힐 가치가 있는가, task-specific tuning 조건은 무엇인가? |

## 옆 저장소와의 접점

`covariate-trust-pilot` 의 `ts-idea-tournament-v1` 브랜치가 이 카탈로그의 두 항목을
이미 실험으로 건드렸다. 같은 주제를 다시 시작하기 전에 그쪽 결과를 먼저 본다.

- **P32 (FAF/AdaRho, AISTATS 2026)** — 그 실험의 Track F 가 AdaRho 를 논문 기준 로컬
  구현으로 재현해 비교했다. 동일 20% 예산에서 AdaRho 가 정상 regime 변화의 72–79% 를
  같이 버렸다. 단 원 논문의 online selection 을 완전히 재현한 것은 아니다.
- **P33 (MTLinear, AISTATS 2025)** — 그 실험의 Track G 가 PCGrad·norm-balanced 대조군을
  두었으나 MTLinear 자체는 read-only 참조로만 두었다.

두 결과 모두 `covariate-trust-pilot` 저장소 `ts-idea-tournament-v1` 브랜치의
`_docs/notes/ts-idea-tournament-v1.md` 에 있다.

## 다음에 할 일

1번 계열(예측 목표 조건부 토큰화)의 1차 파일럿이 `INCONCLUSIVE`로 끝나
[노트의 §12.8 다음 행동 후보](notes/tsfm_topics/01_initial_topics/01_forecast_query_tokenization.md)에
선택지가 정리돼 있다 — 이 계열을 계속할지 접을지는 아직 정해지지 않았다.

2·3번 계열이나 dossier 의 5개 후보 연구군으로 넘어가는 경우, 연구군을 고르기 전에
dossier 의 "비교의 필수 구분" 절을 먼저 읽는다.

### 2026-09-08 문제 중심 PEFT 사전조사18

[사전조사](reference/time_series_peft_problem_first_dossier_20260908.md)와 [후보 결정 노트](notes/tsfm_topics/07_research_direction/18_peft_problem_first_candidates_20260908.md)를 추가했다. 직접 선행·공식 자료29항목을 정리했고, 다음 우선 검사는 실제 revision 정답을 이용한 적응 교정(R1)의 자료 진입이다. 현재-vintage head·발표 차수 보정·standard LoRA replay/reset이 충분하면 새 PEFT 안을 중단한다. R2 집계+소량 fine 감독과 R3 측정/채널 변화는 조건부 보류다. 이번에는 새 학습·17번 사후 대조 실행이 없고, 신규 방법·PEFT 필요성·신규성은 미확정이다. 이전17의 동결 결과를 변경하지 않았다.

### 2026-09-08 수정 정답 R1 CPU 진단19

[결과19](notes/tsfm_topics/05_revision/19_peft_revision_entry_results_20260908.md): ALFRED PAYEMS/INDPRO의실제revision자료를확보하고원자료·causalcoverage를검증했다. CPU8대조를V36/E60origin씩평가했다. REVISED대FIRST평균15.887%개선으로사전screen은통과했으나PAYEMS의λ선택차이가섞였다. 사후동일λ에서는PAYEMS1.726~3.122%악화/INDPRO0.120%개선이며ZERO가전체E두계열에서최저다. 현재두계열의새PEFT GPU진입은보류하고원screenPASS는보존한다. 독립CSV감사PASS·17검사/4subtestsPASS·신규GPU학습0. 실제수정조건의존재를확인한것이며새PEFT방법/필요성을확보한것은아니다.
