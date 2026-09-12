"""Render the completed diagnostic's measured results without changing selection."""
from pathlib import Path
from datetime import datetime
import csv
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
NOTES = ROOT/'_docs/notes/tsfm_topics/07_research_direction'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    s, a, r = [read(OUT/f'{name}.json') for name in ('summary', 'integrity_audit', 'resource_summary')]
    assert s['status'] == 'COMPLETED_DEVELOPMENT_DIAGNOSTIC' and a['status'] == 'PASS'
    guards = [read(p) for p in (ROOT/'runs/peft_lora_only_diagnostic_v1').glob('**/guard/status.json')]
    run = ROOT/'runs/peft_lora_only_diagnostic_v1'
    fit_records = {p.parts[-3]: read(p) for p in run.glob('fit/*/output/result.json')}
    (OUT/'fit_records.json').write_text(json.dumps(fit_records, indent=2)+'\n', encoding='utf-8')
    resource_samples = [json.loads(line) for p in run.glob('**/guard/resource_log.jsonl') for line in p.read_text().splitlines()]
    (OUT/'guard_records.json').write_text(json.dumps({'guards': guards, 'samples': resource_samples}, indent=2)+'\n', encoding='utf-8')
    prepared = read(OUT/'prelaunch_plan.json')['created_utc']
    finished = max(g['finished_at'] for g in guards)
    r['wall_from_initial_plan_to_last_forecast_seconds'] = (datetime.fromisoformat(finished)-datetime.fromisoformat(prepared)).total_seconds()
    r['timing_scope'] = 'Initial plan creation through last forecast; includes admission wait/cleanup, excludes earlier source audit and later reporting.'
    (OUT/'resource_summary.json').write_text(json.dumps(r, indent=2)+'\n', encoding='utf-8')
    plan = read(ROOT/'experiments/peft_lora_only_diagnostic_v1/plan.json')
    with (OUT/'selected_models.csv').open(newline='', encoding='utf-8') as f:
        winners = list(csv.DictReader(f))
    for winner in winners:
        winner['lora_lr'] = plan['jobs'][int(winner['index'])]['lora_lr']
    with (OUT/'selected_models.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(winners[0]))
        writer.writeheader()
        writer.writerows(winners)
    a['trainable_categories'] = {'LoRA': 1179648, 'native_backbone': 0, 'native_output_head': 0,
                                 'residual_MLP': 0, 'other': 0}
    (OUT/'integrity_audit.json').write_text(json.dumps(a, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    lines = ['# Study38 — native-path LoRA-only diagnostic', '',
             f"[판정] 전체 CASE {s['overall_case']}. 이미 노출된 BMRA/Jena Study35 개발 분할의 진단이며 independent/final test가 아니다.", '',
             f"[확인] LoRA-only와 JOINT가 모두 F0보다 나쁜 cell/view는 {sum(x['LORA_ONLY'] > x['F0'] and x['JOINT'] > x['F0'] for x in s['rows'])}/8이다. LoRA-only는 JOINT보다 {sum(x['LORA_ONLY'] < x['JOINT'] for x in s['rows'])}/8에서 좋다. BMRA seed30001 S180에서는 JOINT가 LoRA-only보다 7.461550%F0 좋으며, 나머지7개에서는 LoRA-only가 더 좋다.", '',
             '[추정] 추가 residual MLP만으로 Study35 D 손해를 설명하기 어렵다. 이번 조건에서는 native-head-frozen LoRA-only도 F0 대비 adaptation value를 얻지 못했다. 새 adapter를 만들기보다 fresh chronological 검증을 우선한다.', '',
             '[확인] 아래 raw D loss는 작은 값이 좋다. S180/L720은 같은 720-step trajectory에서 V로 선택한 두 view이며 독립 반복이 아니다.', '',
             '```text', 'dataset seed   view   F0          HEAD        WIDE        LORA_ONLY   JOINT       CASE']
    for row in s['rows']:
        lines.append(f"{row['dataset']:7} {row['seed']} {row['budget']:5} " + ' '.join(f'{row[k]:.9f}' for k in ('F0', 'HEAD', 'WIDE', 'LORA_ONLY', 'JOINT')) + f"  {row['case']}")
    lines += ['```', '', '[확인] 아래 대비는 모두 %F0. G는 F0 대비, I는 WIDE 대비, H_EFFECT는 100(LoRA-only−JOINT)/F0이며 양수면 JOINT가 더 좋다.', '',
              '```text', 'dataset seed   view   G_LORA    G_JOINT   I_LORA    I_JOINT   H_EFFECT']
    for row in s['rows']:
        lines.append(f"{row['dataset']:7} {row['seed']} {row['budget']:5} " + ' '.join(f'{row[k]:+9.4f}' for k in ('G_LORA_ONLY', 'G_JOINT', 'I_LORA_ONLY', 'I_JOINT', 'H_EFFECT')))
    lines += ['```', '', '[판정] CASE A는 LoRA-only가 F0보다 좋고 JOINT 이상, B는 JOINT가 F0와 LoRA-only보다 좋음, C는 둘 다 F0보다 나쁨을 뜻한다. D는 앞 조건에 해당하지 않으면서 다섯 arm 모두 기존0.25%F0 band 안인 경우다. 어느 조건에도 맞지 않으면 UNRESOLVED로 남긴다. 서로 다른 CASE가 나타나면 전체 E로 표기한다.', '',
              '[추정] A cell은 추가 residual MLP가 필요하지 않았거나 공동 적응이 불리했을 가능성과 양립한다. C cell의 손해는 추가 head만으로 설명하기 어렵다. 이 비교는 학습 절차 간 차이이며 representation 복원·정보 생성·표현 부족을 직접 측정하지 않는다. LORA_ONLY의 학습 파라미터 수는 WIDE/JOINT보다 적다.', '',
              '[확인] 기존 Study35 L720 재검산 평균: JOINT−HEAD 개선 +12.134632%F0, JOINT−WIDE 개선 +8.804323%F0, JOINT−F0 개선 −4.304809%F0. 기존 결과 재학습은 없다.', '',
              f"[확인] Step0 F0 max error={a['step0_max_abs_error']}; 실제 학습 파라미터={a['trainable_counts']}; 96 attention LoRA modules/192 tensors. Native head와 backbone 원래 가중치 frozen, residual probe 없음. 학습 후 frozen hash 및 선택 checkpoint 재복원/예측 일치 검사를 통과했다.", '',
              f"[확인] 기존 Study35 소스·입력 해시 {a['old']['study35_hash_count']}개, closure provenance {a['old']['closure_hash_count']}개, 기존 raw D {a['old']['old_rows_recomputed']}행 및 V-only 선택 재검산 통과. 새 plan SHA256 `{a['plan_sha256']}`; selection SHA256 `{a['selection_sha256']}`. metric/target/scale/quantile/origin과 native F0 D 예측이 기존 증거와 일치한다.", '',
              '[확인] LR [1e-5,3e-5,1e-4,3e-4]는 요청한 네 default와 기존 HEAD/WIDE numeric grid를 사용했다. Study35 JOINT LoRA rate는 [1e-5,3e-5,3e-5,1e-4]였으므로 동일한 JOINT grid라고 주장하지 않는다. D를 보고 grid나 threshold를 변경하지 않았다.', '',
              f"[확인] 실행: fit {r['fit_count']}회, 선택 D forecast {r['forecast_count']}회, zero-update smoke {r['zero_update_gate_count']}회; guard {r['guard_count']}회 정상 종료. 최초 plan부터 마지막 forecast까지 wall {r['wall_from_initial_plan_to_last_forecast_seconds']:.2f}s(대기·정리 포함, 이전 소스 감사·후속 보고 제외), 학습/평가 controller wall {r['completed_run_wall_seconds']:.2f}s, fit worker wall 합 {r['fit_wall_seconds_sum']:.2f}s, 학습 trajectory 합 {r['fit_trajectory_seconds_sum']:.2f}s.", '',
              f"[확인] Peak CUDA allocated {r['peak_gpu_allocated_bytes']/2**20:.1f}MiB / reserved {r['peak_gpu_reserved_bytes']/2**20:.1f}MiB; sampled device 전체 메모리 {r['sampled_device_memory_mib']:.1f}MiB. Runtime 최소 여유 RAM {r['min_available_ram_gib']:.2f}GiB, commit {r['min_available_commit_gib']:.2f}GiB. 장치 sampled memory와 PyTorch allocator peak는 서로 다른 측정이다.", '',
              '[확인] 최초 준비에서 로그 폴더 생성 누락으로 GPU 실행 전 FileNotFoundError가 발생했다. 최초 plan과 traceback을 보존하고 폴더 생성 순서만 수정한 뒤 재봉인했다. 메모리 기준을 낮추지 않았으며, 사용자 승인으로 부모가 종료된 watcher/helper와 미사용 자동화 도우미, 오래된 유휴 Claude CLI를 정리했다. 학습·원천·sealed 결과 파일은 삭제하지 않았다.', '',
              '[판정] Fresh Stage A: BDG2 Bull Office 2016-01-01~2016-08-10 READY(개발 후보); Household post-P1 BLOCKED(48h target 결측 창). Household는 성능을 보지 않은 origin exclusion/gap 또는 split 정책을 먼저 정해야 한다. 두 후보 pretraining overlap은 UNKNOWN. Fresh 학습/forecast는0회다.', '',
              '[확인] Time-PEFT 원문 p.6 §5.1.3, p.12 Algorithm1, p.16 §D.1.3에서 forecast-head 공동 학습을 확인했다. p.16 §D.2.2는 native architecture 보존 여부와 손해 차이도 논한다. 해당 경계는 별도 Study37 문서에 갱신했으며 PDF는 commit하지 않는다.', '',
              '[판정] 새 adapter 개발이나 READY_FOR_PAPER 승격은 하지 않는다. 결과를 검토한 뒤 fresh F0/HEAD/WIDE/LORA_ONLY/JOINT 계약을 별도로 봉인해야 한다. 이번 실행은 main commit/push로 끝낸다.']
    body = '\n'.join(lines)+'\n'
    (OUT/'STATUS.md').write_text(body+'\n[확인] 상세: [metrics](metrics.csv), [selected models](selected_models.csv), [integrity](integrity_audit.json), [resources](resource_summary.json).\n\n![D/F0](figures/01_loss_ratios.png)\n\n![Contrasts](figures/02_contrasts.png)\n', encoding='utf-8')
    (NOTES/'38_lora_only_diagnostic_20260912.md').write_text(body+'\n[확인] [결과와 그림](../../../../results/peft_lora_only_diagnostic_v1/STATUS.md) · [fresh manifest](../../../../results/peft_paper_closure_v1/fresh_stage_a_candidate_manifest.md) · [선행 경계](37_novelty_boundary.md)\n', encoding='utf-8')


if __name__ == '__main__':
    main()
