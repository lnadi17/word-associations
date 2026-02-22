from __future__ import annotations

import random
from typing import List


def sample_cues(cues: List[str], num_cues: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    if num_cues >= len(cues):
        return list(cues)
    return rng.sample(cues, num_cues)
