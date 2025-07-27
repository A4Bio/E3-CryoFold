import torch
import torch.nn as nn
import esm
from chroma.models.graph_backbone import GraphBackbone

esm_model,alphabet = esm.pretrained.esm2_t12_35M_UR50D()

class Model(nn.Module):
    def __init__(self, encoder_dim=512, dropout=0.1,init_kwargs=None):
        super().__init__()
        self.dropout = dropout
        self.esm = esm_model
        self.encode_head = nn.Linear(480, encoder_dim)

        self.encoder = GraphBackbone(**init_kwargs)
        self.position_emb = nn.Embedding(num_embeddings=30000, embedding_dim=encoder_dim)
        self.chain_embed = nn.Embedding(num_embeddings=1000, embedding_dim=encoder_dim, padding_idx=0)

    
    def infer(self, X, 
                chain_encoding, 
                seqs=None,
                t=1.0
                ):
        
        device = X.device
        if isinstance(t, float):
            t = torch.FloatTensor([t]).to(device)
        seq_pos = torch.arange(chain_encoding.shape[0]).to(device)[None, :]
        chain_encoding = chain_encoding.long()[None,:].to(device)

        if chain_encoding.shape[1] != X.shape[1]:
            imputed_X = torch.zeros(list(chain_encoding.shape) + [4, 3]).to(device) + X.mean(1, keepdim=True)
            imputed_X[:, :chain_encoding.shape[1]] = X[:, :chain_encoding.shape[1]]

        with torch.no_grad():
            h_V = self.get_h_V(seq_pos=seq_pos, chain_encoding=chain_encoding, seqs=seqs)
            X_pred = self.encoder(imputed_X, chain_encoding, t, h_V, if_infer=True)
        return  X_pred
    
    def get_h_V(self,
                seq_pos, 
                chain_encoding, 
                seqs=None
                ):
        device = chain_encoding.device
        length = len(seqs)
        seqs = torch.LongTensor(alphabet.encode(''.join(seqs))).to(device)[None,:]

        h_Vs = []
        for i in range(0, length, 3500):
            start, end = i, i+3500
            local_len = min(3500, length-start)
            if i + 3500 > length:
                start = max(length-3500, 0)

            h_V = self.esm(seqs[:,start: end], repr_layers=[12])['representations'][12] 
            h_Vs.append(h_V[:, -local_len:])

        h_V = torch.cat(h_Vs, dim=1)
        h_V = self.encode_head(h_V)
        h_V = h_V + self.chain_embed(chain_encoding) + self.position_emb(seq_pos)

        return  h_V
