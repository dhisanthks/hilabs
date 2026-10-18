"""HiLabs clinical term-mapping engine — single entry point.

  python run.py train                         # build all models from hackathon_train.csv (resumable)
  python run.py predict --input X.csv --output preds.csv
  python run.py all --input X.csv --output preds.csv   # train if needed, then predict
"""
import argparse
import os
import sys
import warnings

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src import pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["train", "predict", "all"])
    ap.add_argument("--train", default=None, help="training CSV (default: hackathon_train.csv)")
    ap.add_argument("--input", help="file to map")
    ap.add_argument("--output", default="outputs/predictions.csv")
    ap.add_argument("--review-threshold", type=float, default=0.5)
    ap.add_argument("--assume-subconcept", default=None,
                    help="fill a missing/empty subconcept column, e.g. DiagnosisCode when the input is known to contain "
                         "only diagnoses/conditions (the brief's scope)")
    ap.add_argument("--no-llm", action="store_true",
                    help="skip the fine-tuned LLM stage (needs a CUDA GPU; automatic when none is found): faster, ~3-4 "
                         "points lower accuracy")
    a = ap.parse_args()
    if a.cmd in ("predict", "all") and not a.input:
        ap.error("--input is required")
    if a.output and os.path.dirname(a.output):
        os.makedirs(os.path.dirname(a.output), exist_ok=True)
    if a.cmd in ("train", "all"):
        if a.cmd == "train" or not os.path.exists(os.path.join(pipeline.ART, "ranker_features.json")):
            pipeline.train(a.train)
    if a.cmd in ("predict", "all"):
        pipeline.predict(a.input, a.output, a.review_threshold, assume_subconcept=a.assume_subconcept,
                         use_llm=not a.no_llm)


if __name__ == "__main__":
    sys.exit(main())
