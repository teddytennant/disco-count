"""Train the tiny neighbor-restricted score network on the 3-node chain."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from disco.estimate import train_score


def main():
    result = train_score(steps=300, seed=0, batch_size=128, width=64)
    loss_start = result["loss_start"]
    loss_end = result["loss_end"]
    print(f"loss_start={loss_start:.8f}")
    print(f"loss_end={loss_end:.8f}")
    if loss_end < 0.9 * loss_start:
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
