"""Prefix-only, prespecified contribution plateau and validation stopping rules."""


def contribution_plateau(history, minimum_step, fraction=.25):
    points = [h for h in history if 'contribution_halves' in h]
    if len(points) < 4 or points[-1]['step'] < minimum_step:
        return False, []
    slopes = [[(b['contribution_halves'][j] - a['contribution_halves'][j]) /
               (b['step'] - a['step']) for j in range(2)]
              for a, b in zip(points, points[1:])]
    thresholds = [fraction * max(0., *[s[j] for s in slopes[:-1]]) for j in range(2)]
    return all(slopes[-1][j] <= thresholds[j] for j in range(2)), [slopes[-1], thresholds]


def early_stop(history, patience=2):
    best, stale = history[0]['score'], 0
    for h in history[1:]:
        if h['score'] < best:
            best, stale = h['score'], 0
        else:
            stale += 1
    return stale >= patience
