import esm
import torch
import torch.nn as nn
from src.graph_module import StructDecoder
from src.transformer_module import Embeddings3D, TransformerBlock, MultiHeadCrossAttention


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout, extract_layers, dim_linear_block):
        super().__init__()
        self.layer = nn.ModuleList()
        self.extract_layers = extract_layers

        self.block_list = nn.ModuleList()
        for _ in range(num_layers):
            self.block_list.append(
                TransformerBlock(dim=embed_dim, heads=num_heads, dim_linear_block=dim_linear_block, dropout=dropout,
                                 prenorm=False))

    def forward(self, x, seq, mask=None):
        for layer_block in self.block_list:
            x, seq = layer_block(x, seq, mask)
        return x, seq


class CryoFold(nn.Module):
    def __init__(self, img_shape=(360, 360, 360), input_dim=1, output_dim=4, embed_dim=768, patch_size=36,
                num_heads=6, dropout=0.1, ext_layers=[3, 6, 9, 12], norm="instance",
                dim_linear_block=3072, decoder_dim=256):
        super().__init__()
        self.num_layers = 8
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.ext_layers = ext_layers
        self.decoder_dim = decoder_dim

        esm_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        self.esm = esm_model

        self.norm = nn.BatchNorm3d if norm == 'batch' else nn.InstanceNorm3d
        self.embed = Embeddings3D(input_dim=input_dim, embed_dim=embed_dim, cube_size=img_shape,
                                  patch_size=patch_size, dropout=dropout)
        self.position_emb = nn.Embedding(num_embeddings=30000, embedding_dim=embed_dim)
        self.token_embed = nn.Embedding(num_embeddings=33, embedding_dim=embed_dim, padding_idx=alphabet.padding_idx)
        self.chain_embed = nn.Embedding(num_embeddings=1000, embedding_dim=embed_dim, padding_idx=0)
        
        self.transformer = TransformerEncoder(embed_dim, num_heads, self.num_layers, dropout, ext_layers,
                                              dim_linear_block=dim_linear_block)
        self.out = nn.Linear(embed_dim, 12)
        self.to_hV = nn.Linear(embed_dim, decoder_dim)
        self.decoder_struct = StructDecoder(8, decoder_dim, 1)
        self.atom_norm = nn.LayerNorm(12)
        self.cross_attn = MultiHeadCrossAttention(embed_dim, num_heads)

    def forward(self, x, seq, seq_pos, chain_encoding, mask=None):
        batch_size = x.shape[0]
        _, length = seq.shape

        transformer_input = self.embed(x)

        ys = []
        h_Vs = []
        for i in range(0, length, 1000):
            seq_feat = self.esm(seq[:, i: i+1000], repr_layers=[12])['representations'][12] 
            seq_feat = seq_feat + self.chain_embed(chain_encoding[:, i: i+1000]) + self.position_emb(seq_pos[:, i:i+1000])
            protein, seq_feat = self.transformer(transformer_input, seq_feat, mask.float()[:, i: i+1000])
            y = self.cross_attn(seq_feat, protein, protein)
            h_V = self.to_hV(y)
            ys.append(y)
            h_Vs.append(h_V)
        y = torch.cat(ys, dim=1)
        h_V = torch.cat(h_Vs, dim=1)

        X = self.atom_norm(self.out(y)).view(batch_size, length, 4, 3)[mask]
        batch_id = torch.arange(x.shape[0]).view(-1, 1).repeat(1, length).to(x.device)[mask]
        chain_encoding = chain_encoding[mask]
        X_pred, all_preds = self.decoder_struct.infer_X(X, h_V[mask], batch_id, chain_encoding, 30, virtual_frame_num=3)
        return X_pred, all_preds
    
    def infer(self, cryo_map, seq, chain_encoding, max_len=30000):
        self.eval()
        seq_pos = torch.arange(seq.shape[0], device=cryo_map.device)
        seq, chain_encoding, seq_pos= map(lambda x: x[:max_len].unsqueeze(0), [seq, chain_encoding.long(), seq_pos])
        protein_data = cryo_map.reshape(1, 1, *cryo_map.shape)
        mask = torch.ones_like(seq).bool()

        X_pred, all_preds = self.forward(protein_data, seq, seq_pos, chain_encoding, mask)
        return X_pred, all_preds
