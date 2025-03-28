import os
import argparse
import torch
from src.chroma.data import Protein
from model import CryoFold
from utils import align, get_data, get_coord_from_pdb


def parse_arguments():
    parser = argparse.ArgumentParser(description="CryoFold: density map to protein structure")
    parser.add_argument('--density_map_path', type=str, help='Path to the input data directory')
    parser.add_argument('--pdb_path', type=str, default=None, help='Path to the ground truth PDB file')
    parser.add_argument('--model_path', type=str, default='pretrained_model/checkpoint.pt', 
                        help='Path to the pretrained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results', 
                        help='Directory to save the output PDB file')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda', 
                        help='Device to run the model on')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output for debugging')
    return parser.parse_args()


def load_model(model_path: str, device: str) -> CryoFold:
    cryofold = CryoFold(
        img_shape=(360, 360, 360), input_dim=1, output_dim=4, embed_dim=480,
        patch_size=36, num_heads=8, dropout=0.1, ext_layers=[3, 6, 9, 12], 
        norm="instance", decoder_dim=128
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    try:
        cryofold.load_state_dict(checkpoint)
    except:
        checkpoint = {k.replace('_forward_module.',''):v for k,v in checkpoint.items()}
        cryofold.load_state_dict(checkpoint)
    return cryofold


def preprocess_data(dir_path: str, device: str = 'cuda'):
    print('Preprocessing data...')
    data = get_data(dir_path)
    maps, seq, chain_encoding = (torch.from_numpy(x).to(device) for x in data)
    print('Preprocessing finished!')
    return maps, seq, chain_encoding


def infer_structure(model: CryoFold, maps: torch.Tensor, seq: torch.Tensor, 
                    chain_encoding: torch.Tensor, coords: torch.Tensor = None):
    try:
        pred_x, _ = model.infer(maps, seq, chain_encoding)
        if coords is not None:
            pred_x, rmsd = align(pred_x, coords)
            print(f'RMSD: {rmsd:.4f}')
    except Exception as e:
        print(f"Error during inference: {e}")
    return pred_x, seq, chain_encoding


def save_protein(preds: torch.Tensor, seqs: torch.Tensor, chain_encodings: torch.Tensor, output_dir: str, output_name: str = 'output'):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name + '.pdb')

    # Process sequence values to ensure they fall within a valid range
    seqs[seqs > 19] = 0
    protein = Protein.from_XCS(preds.unsqueeze(0), chain_encodings.unsqueeze(0), seqs.unsqueeze(0))
    protein.to_PDB(output_path)
    print(f"Protein structure saved to {output_path}")


def main():
    args = parse_arguments()
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    
    if args.verbose:
        print(f"Running on device: {device}")

    model = load_model(args.model_path, device)
    maps, seq, chain_encoding = preprocess_data(args.density_map_path, device)

    coords = None
    if args.pdb_path:
        coords = get_coord_from_pdb(args.pdb_path)
        coords = torch.from_numpy(coords).to(device)

    preds, seqs, chain_encodings = infer_structure(model, maps, seq, chain_encoding, coords)
    save_protein(preds, seqs, chain_encodings, args.output_dir)

if __name__ == "__main__":
    main()
