"""Prospective, validation-only choices; no evaluation outcomes are accepted."""
import hashlib
import numpy as np

ACTIONS = ('STOP', 'HEAD', 'JOINT')


def probe_choice(scores, initial_score, margin_pct=0.):
    best = min(scores.values())
    tolerance = margin_pct * initial_score / 100.
    return next(action for action in ACTIONS if action in scores and scores[action] <= best + tolerance)


def random_choice(key):
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
    return ACTIONS[int(np.random.default_rng(seed).integers(3))]


def selected(history, patience=None):
    best, stale = history[0], 0
    for point in history[1:]:
        if point['V'] < best['V']:
            best, stale = point, 0
        else:
            stale += 1
        if patience is not None and stale >= patience:
            return best, point['step']
    return best, history[-1]['step']


def paths(result, margin_pct=0., constant='STOP'):
    histories = result['histories']
    fork = result['fork']
    joint = histories['JOINT']
    prefix = [p for p in joint if p['step'] <= fork]
    qstep = fork + result['probe_steps']
    at = lambda history, step: next(p for p in history if p['step'] == step)
    trial = {'STOP': at(joint, fork)['V'],
             'HEAD': at(histories['HEAD1'], qstep)['V'],
             'JOINT': at(joint, qstep)['V']}
    action = probe_choice(trial, joint[0]['V'], margin_pct)
    key = '/'.join(str(result['job'][k]) for k in ('episode', 'dataset', 'condition', 'seed'))
    choices = {
        'PROBE': action,
        'CURRENT_C': 'JOINT' if result['current_C_pct_F0'] > 0 else 'HEAD',
        'RANDOM': random_choice(key),
        'HEAD_PROBE': probe_choice({a:trial[a] for a in ('STOP','HEAD')},joint[0]['V'],margin_pct),
        'CONSTANT': constant,
    }
    options = {'STOP': prefix, 'HEAD': histories['HEAD1'], 'JOINT': joint}
    output = {name:{'action':act,'selected':selected(options[act])[0],
                    'final':options[act][-1]} for name,act in choices.items()}
    for mode in ('JOINT', 'HEAD0', 'HEAD1', 'HEAD2', 'MASKED1'):
        history = histories[mode]
        output[mode] = {'action':mode,'selected':selected(history)[0],'final':history[-1]}
    for patience in (1,2,3):
        point, stop = selected(joint, patience)
        output[f'ES{patience}'] = {'action':f'ES{patience}','selected':point,
                                 'final':at(joint,stop),'stop_step':stop}
    output['STOP'] = {'action':'STOP','selected':selected(prefix)[0],'final':prefix[-1]}
    return output
