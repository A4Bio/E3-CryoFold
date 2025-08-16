"""
author: nabin 
timestamp: Tue Jan 02 2024 02:00 PM
"""

import warnings
warnings.filterwarnings('ignore')
import pytorch_lightning as pl
from cryo_module.AutoEM import run_pulchra

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

import numpy as np
from argparse import ArgumentParser
from E3CryoFold2 import E3CryoFold2

        

def main():
    pl.seed_everything(42)
    parser = ArgumentParser()
    parser = E3CryoFold2.add_model_specific_args(parser)
    # training specific args

    parser.add_argument('--pretrained', type=str, default='./models/model_weight.pth')
    parser.add_argument('--stru_pretrained', type=str, default='./models/structure_model.pth', help='structure model; it is needed when user choose the protocol of denovo')
    parser.add_argument('--map_path', type=str, default='', help='path of EM map')
    parser.add_argument('--fasta_path', type=str, default='')
    parser.add_argument('--protocol', type=str, default='pre_align', choices=['pre_align', 'denovo', 'seq_free', 'seq_free_denovo'])
    parser.add_argument('--save_dir', type=str, default='./data/outputs/')
    parser.add_argument('--save_name', type=str, default='8623-5uz7-prealign-0.1')
    parser.add_argument('--spatial_condition', type=bool, default=True, help='whether use spatial conditioner')
    parser.add_argument('--sequence_condition', type=bool, default=False, help='whether use sequence conditioner')
    parser.add_argument('--t', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    model = E3CryoFold2(args=args, model_weights=args.pretrained, struc_model_weight=args.stru_pretrained).to(args.device)

    protein = model.infer(t=args.t, spatial_conditioner=args.spatial_condition, sequence_conditioner=args.sequence_condition)
    protein.to_PDB(args.save_dir + args.save_name + '.pdb')
    
    if args.pulchra:
        run_pulchra(args.save_dir, args.pulchra_path, args.save_dir+ args.save_name + '.pdb', args.save_name)


if __name__ == "__main__":
    main()
