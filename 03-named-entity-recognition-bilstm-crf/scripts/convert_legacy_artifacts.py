"""Document and normalize the original model and mapping filenames.

This script does not claim to convert the original softmax model into a trained
CRF. It only copies artifacts and records architecture metadata honestly.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--word2idx", type=Path, required=True)
    parser.add_argument("--tag2idx", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.model, args.destination / "legacy_bilstm_softmax_model.h5")
    shutil.copy2(args.word2idx, args.destination / "word_to_index.pkl")
    shutil.copy2(args.tag2idx, args.destination / "tag_to_index.pkl")
    with args.tag2idx.open("rb") as handle:
        tag_to_index = pickle.load(handle)
    with (args.destination / "index_to_tag.pkl").open("wb") as handle:
        pickle.dump({index: tag for tag, index in tag_to_index.items()}, handle)
    print(json.dumps({"destination": str(args.destination), "is_true_crf": False}, indent=2))


if __name__ == "__main__":
    main()
