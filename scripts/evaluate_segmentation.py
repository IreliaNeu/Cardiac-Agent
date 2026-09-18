"""Evaluate official EchoNet weights against CAMUS manual LV annotations."""
import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from cardiac_agent.benchmark import evaluate_studies
from cardiac_agent.explanation import QwenClient
from cardiac_agent.pipeline import digest
from cardiac_agent.roi import EchoNetSegmenter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--studies", nargs="+", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qwen", action="store_true")
    parser.add_argument("--env", type=Path)
    parser.add_argument("--deformation", choices=["none", "intensity_difference", "farneback"],
                        default="intensity_difference")
    args = parser.parse_args()
    if args.env:
        if not args.env.is_file():
            parser.error("Environment file does not exist")
        load_dotenv(args.env)
    config = json.loads(args.config.read_text())
    checkpoint = Path(config["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = (args.config.parent / checkpoint).resolve()
    import torch
    import torchvision
    torch.set_num_threads(2)
    segmenter = EchoNetSegmenter(checkpoint, config["mean"], config["std"], config["device"])
    provenance = {"backend": "echonet_deeplabv3", "checkpoint_sha256": digest(checkpoint),
                  "config_sha256": digest(args.config), "config": config,
                  "torch": torch.__version__, "torchvision": torchvision.__version__}
    result = evaluate_studies(args.studies, args.output, segmenter, provenance,
                              QwenClient() if args.qwen else None, args.deformation)
    print(json.dumps(result, indent=2))
    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
