import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="")

    # datasets
    parser.add_argument('--root_dir', type=str, default='../datasets')
    parser.add_argument('--obs_dir', type=str, default='../observations_img_v1')
    parser.add_argument('--dataset', type=str, default='r2r', choices=['r2r', 'r4r'])
    parser.add_argument('--output_dir', type=str, default='../datasets/R2R/exprs/NavGPT-Flant5XL-unseen0', help='experiment id')
    parser.add_argument('--seed', type=int, default=0)

    # Agent
    parser.add_argument('--temperature', type=float, default=0.0, help='temperature for llm')
    # parser.add_argument('--llm_model_name', type=str, default='gpt-3.5-turbo', help='llm model name')
    # parser.add_argument('--llm_model_name', type=str, default='gpt-4', help='llm model name')
    # parser.add_argument('--llm_model_name', type=str, default='LlaMA-2-7b-chat', help='llm model name')
    # parser.add_argument('--llm_model_name', type=str, default='Emu-14B', help='llm model name')
    parser.add_argument('--llm_model_name', type=str, default='NavGPT_InstructBLIP', help='llm model name')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_iterations', type=int, default=10)
    parser.add_argument('--action_space', type=str, default='angle', choices=['angle', 'viewpoint'])

    # Model config
    # parser.add_argument("--instruct", action='store_true', default=True, help="Load Emu-I")
    # parser.add_argument("--ckpt_path", type=str, default='/raid/data/weight/withoutmap.pt', help="Emu ckpt path")
    parser.add_argument("--ckpt_path", type=str, default='..im/checkpoint_best.pth', help="Instructnav ckpt path")
    parser.add_argument("--num_beams", type=int, default=5, help="num beams")
    parser.add_argument("--max_length", type=int, default=512, help="max length")
    parser.add_argument("--min_length", type=int, default=1, help="min length")

    # General config
    # parser.add_argument('--iters', type=int, default=10, help='number of iterations to run')
    parser.add_argument('--iters', type=int, default=None, help='number of iterations to run')
    parser.add_argument('--max_scratchpad_length', type=int, default=1000, help='max number of steps in an episode')
    parser.add_argument('--test', action='store_true', default=False)
    # parser.add_argument('--val_env_name', type=str, default='R2R_val_72_instr_0')
    # parser.add_argument('--val_env_name', type=str, default='R2R_val_72_instr_1')
    # parser.add_argument('--val_env_name', type=str, default='R2R_val_72_instr_2')
    parser.add_argument('--val_env_name', type=str, default='R2R_val_unseen_instr_0')
    # parser.add_argument('--val_env_name', type=str, default='R2R_val_unseen_instr_1')
    # parser.add_argument('--val_env_name', type=str, default='R2R_val_unseen_instr_2')
    # parser.add_argument('--val_env_name', type=str, default='R2R_val_unseen_instr_3')
    # parser.add_argument('--val_env_name', type=str, default='R2R_val_unseen_instr_4')
    # parser.add_argument('--val_env_name', type=str, default='R2R_val_unseen_instr')

    parser.add_argument('--detailed_output', action='store_true', default=True)

    # parser.add_argument('--valid_file', type=str, default='../datasets/R2R/exprs/4-R2R_val_unseen_instr/4-R2R_val_unseen_instr.json', help='valid file name')
    parser.add_argument('--valid_file', type=str, default=None, help='valid file name')

    args, _ = parser.parse_known_args()

    args = postprocess_args(args)

    return args


def postprocess_args(args):
    ROOTDIR = args.root_dir

    # Setup input paths
    # args.obs_dir = os.path.join(ROOTDIR, 'R2R', 'observations_v1')
    args.obs_summary_dir = os.path.join(ROOTDIR, 'R2R', 'observations_summarized')
    args.map_dir = None

    args.connectivity_dir = os.path.join(ROOTDIR, 'R2R', 'connectivity')
    args.scan_data_dir = os.path.join(ROOTDIR, 'Matterport3D', 'v1_unzip_scans')

    args.anno_dir = os.path.join(ROOTDIR, 'R2R', 'annotations')
    args.navigable_dir = os.path.join(ROOTDIR, 'R2R', 'navigable')

    # Build paths
    args.log_dir = os.path.join(args.output_dir, 'logs')
    args.pred_dir = os.path.join(args.output_dir, 'preds')

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.pred_dir, exist_ok=True)

    return args

