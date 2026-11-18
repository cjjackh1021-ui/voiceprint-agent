"""Grade agent decisions against hidden labels.

The agent's similarity score is not the label. Uncertain is not counted as a correct automatic decision.
"""


def grade(expect: str, decision: str | None) -> dict:
    decision = decision or 'missing'
    return {
        'correct': decision == expect,
        'false_accept': expect == 'reject' and decision == 'accept',
        'false_reject': expect == 'accept' and decision == 'reject',
        'reviewed': decision == 'uncertain',
        'delivered': decision in ('accept', 'reject', 'uncertain', 'unknown'),
        'decision': decision,
        'expect': expect,
    }


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {'n': 0, 'accuracy': None, 'strict_accuracy': None, 'auto_accuracy': None,
                'false_accept_rate': None, 'false_reject_rate': None,
                'review_rate': None, 'delivery_rate': None}
    def rate(key, chosen=None):
        pool = rows if chosen is None else [row for row in rows if row.get('expect') == chosen]
        if not pool:
            return None
        return sum(bool(row[key]) for row in pool) / len(pool)
    decided = [row for row in rows if not row.get('reviewed') and row.get('decision') != 'missing']
    auto = None if not decided else sum(bool(row['correct']) for row in decided) / len(decided)
    return {
        'n': n,
        'accuracy': rate('correct'),
        'strict_accuracy': rate('correct'),
        'auto_accuracy': auto,
        'false_accept_rate': rate('false_accept', 'reject'),
        'false_reject_rate': rate('false_reject', 'accept'),
        'review_rate': rate('reviewed'),
        'delivery_rate': rate('delivered'),
    }


def paired_bootstrap(base: list[bool], cand: list[bool], draws: int = 2000, seed: int = 0) -> dict:
    import numpy as np
    left = np.asarray(base, dtype=np.float64)
    right = np.asarray(cand, dtype=np.float64)
    if left.shape != right.shape or left.size == 0:
        raise ValueError('paired comparison needs the same non-empty cases')
    diff = float(right.mean() - left.mean())
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, left.size, size=(draws, left.size))
    samples = right[pick].mean(axis=1) - left[pick].mean(axis=1)
    low, high = (float(x) for x in np.quantile(samples, [0.025, 0.975]))
    return {'diff': diff, 'low': low, 'high': high, 'n': int(left.size)}
