# parse_args.py
import argparse
import os

def dataset_list(arg):
    return arg.split(',')

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an MLLM-powered VLN agent on MP3D-based tasks"
    )

    # 1. Task / dataset
    parser.add_argument(
        "--datasets",
        type=dataset_list,
        default=["r2r"],
        required=True,
        help="Which VLN task(s) to run (comma-separated list)"
    )
    
    parser.add_argument(
        "--splits",
        type=dataset_list,
        default=["val_unseen_subset"],
        help="Split to evaluate (e.g. val_seen, val_unseen)"
    )

    # 2. Hugging Face repos (we’ll download JSONs directly)
    parser.add_argument(
        "--anno_hf_repo",
        type=str,
        default="billzhao1030/vln-annotations",
        help="HF repo ID containing instruction annotations"
    )
    parser.add_argument(
        "--mp3d_hf_repo",
        type=str,
        default="billzhao1030/MP3D",
        help="HF repo ID containing navigable.json & location.json"
    )

    # 3. Precomputed feature stores (still local)
    parser.add_argument(
        "--obs_dir",
        type=str,
        default="../datasets/observations/",
        help="Local directory of image files"
    )
    parser.add_argument(
        "--obs_sum_dir",
        type=str,
        default="../datasets/observations_summary/",
        help="Local directory of summary of image files"
    )
    parser.add_argument(
        "--map_dir",
        type=str,
        default=None,
        help="(Optional) Local directory of map features"
    )
    parser.add_argument(
        "--obj_feat_dir",
        type=str,
        default=None,
        help="(Optional) Local LMDB file of object features"
    )

    # 4. Batch & randomness
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of parallel envs / batch size")
    parser.add_argument("--seed",       type=int, default=0,
                        help="Random seed for shuffling")

    # 5. Output & logging
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Base directory for logs & predictions")
    parser.add_argument("--model_name", type=str, default="qwen2_5_vl",
                        help="Name of the MLLM agent, e.g. llava-v1.5")

    # 6. (Optional) local cache for downloaded JSONs
    parser.add_argument("--cache_dir", type=str, default="~/.cache/vln",
                        help="Where to cache HF-downloaded JSON files")

    args = parser.parse_args()

    # build output subdirs
    args.log_dir  = os.path.join(args.output_dir, "logs")
    args.pred_dir = os.path.join(args.output_dir, "preds")
    for d in (args.output_dir, args.log_dir, args.pred_dir):
        os.makedirs(os.path.expanduser(d), exist_ok=True)

    return args
