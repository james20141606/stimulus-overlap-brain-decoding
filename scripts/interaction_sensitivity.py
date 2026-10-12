#!/usr/bin/env python3
"""Review-time sensitivity only; t/flip assumptions do not certify the original design."""
import itertools
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'results/interaction_receipt.json'
receipt = json.loads(SOURCE.read_text())
contrasts = {}
for budget, block in receipt['budgets'].items():
    values = {name: value['per_subject'] for name, value in block['contrasts'].items()
              if name in ('split_main', 'guard_main')}
    values['split_minus_guard'] = [a-b for a, b in zip(values['split_main'], values['guard_main'])]
    for name, array in values.items():
        n = len(array)
        mean = statistics.mean(array)
        se = statistics.stdev(array) / math.sqrt(n)
        exceed = sum(abs(sum(v*s for v, s in zip(array, signs))/n) >= abs(mean)-1e-12
                     for signs in itertools.product((-1, 1), repeat=n))
        contrasts[f'{budget}_{name}'] = {
            'mean': mean, 'se': se,
            't9_ci95': [mean-2.262157*se, mean+2.262157*se],
            'two_sided_signflip_p': exceed / 2**n,
            'positive': sum(v > 0 for v in array), 'n': n,
        }
result = {
    'note': 'Review-only sensitivity from shipped 2026-09-21 receipt; t intervals assume independent subject contrasts; sign flips assume symmetric null. Not pre-registered, not a replacement confirmatory result. No new training.',
    'source': 'results/interaction_receipt.json', 'contrasts': contrasts,
}
print(json.dumps(result, indent=2))
