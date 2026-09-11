"""Prepare new checked development periods using the unchanged archive contract."""
from datetime import datetime
from pathlib import Path
import json
from experiments.peft_initial_headroom_v1 import prepare as previous

ROOT = Path(__file__).resolve().parents[2]
STUDY = 'peft_head_convergence_v1'


def prepare(output):
    previous.STUDY = STUDY
    previous.VERSION = STUDY+'.20260911'
    previous.EPISODE_STARTS = {'diag': {'jena': datetime(2018,5,4), 'bmra': datetime(2017,5,4)}}
    previous.JENA_FILES = {'diag': tuple(ROOT/f'data/jena_mpi_roof/mpi_roof_2018{s}.csv' for s in ('a','b'))}
    previous.base.BMRA_CHANNELS = ('E_BRYBW-1','E_BURBO','E_DALSW-1','E_BNWKW-1')
    previous.KNOWN_EXPOSURES = {k: list(v) for k,v in previous.KNOWN_EXPOSURES.items()}
    previous.KNOWN_EXPOSURES['jena'].append({'study':'hq_token_pilot_v1/weather and FEV jena_weather_1D',
        'path':'results/hq_token_pilot_v1/data_manifest.json','start':'2020-01-01T00:00:00',
        'end_exclusive':'2021-01-01T00:10:00','label_use':'Earlier weather2020 fit/validation/test; same raw readings as MPI2020'})
    for dataset, start, end in [('jena','2019-05-04T00:00:00','2020-01-01T00:00:00'),('bmra','2018-01-04T00:00:00','2018-09-03T00:00:00')]:
        previous.KNOWN_EXPOSURES[dataset].append({'study':'peft_initial_headroom_v1','path':'runs/peft_initial_headroom_v1/prepared/summary.json','start':start,'end_exclusive':end,'label_use':'Study34 development diagnostic'})
    # Replace only generated source notes, not the frozen previous files or archive semantics.
    original_replace = previous.replace
    def current_notes(spec, **kwargs):
        notes = ('Two fixed targets and two context-only covariates; native hourly archive contract.',
                 'Study35 development period: Jena2018 or BMRA2017 May4; source/pretraining exposure not excluded.',
                 'BMRA same four series but new target pair chosen by prespecified-window availability before model outcomes; not a same-unit replication.')
        additions={'source_notes':notes}
        if spec.name=='jena':
            additions['download_url']='https://www.bgc-jena.mpg.de/wetter/mpi_roof_2018a.zip ; https://www.bgc-jena.mpg.de/wetter/mpi_roof_2018b.zip'
        return original_replace(spec, **(kwargs | additions))
    previous.replace = current_notes
    original_bmra = previous.base.load_bmra
    def current_bmra(episode):
        spec, timestamps, values, metadata = original_bmra(episode)
        metadata['selection_rule'] = 'Same four Study33 series, rotated target roles before outcomes after original E_BNWKW-1 target failed70% window coverage; targets E_BRYBW-1/E_BURBO, covariates E_DALSW-1/E_BNWKW-1. Proposed-period availability conditions this choice.'
        return spec, timestamps, values, metadata
    previous.base.load_bmra = current_bmra
    original_write = previous._write_json_x
    def current_write(path, payload):
        if 'exposure_summary' in payload:
            payload['exposure_summary']['selection_limit'] = 'BMRA keeps four Study33 series but rotates targets to E_BRYBW-1/E_BURBO after 2017 E_BNWKW-1 coverage rejection; raw availability-only choice across proposed splits before any model outcomes. Different unit-pair and season, not same-unit replication.'
        return original_write(path, payload)
    previous._write_json_x = current_write
    try:
        return previous.prepare(output)
    finally:
        previous.replace = original_replace
        previous._write_json_x = original_write
        previous.base.load_bmra = original_bmra


if __name__ == '__main__':
    print(json.dumps(prepare(ROOT/'runs'/STUDY/'prepared'),indent=2))
