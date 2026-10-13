"""Step 3: train the expert model and export its logits on the memory and test ECGs (4 GPUs, about one hour).

  torchrun --standalone --nproc_per_node 4 scripts/03_train_expert.py

Writes work/expert/best.pt, logits_{memory,test}.npy and ids_{memory,test}.json.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadre.expert.train import main  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256, help="per GPU")
    ap.add_argument("--lr", type=float, default=2e-3)
    a = ap.parse_args()
    main(epochs=a.epochs, batch=a.batch, lr=a.lr)
