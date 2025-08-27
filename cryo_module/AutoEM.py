import os
import mrcfile
import numpy as np
import cryo_module.utils as utils
import torch
import open3d as o3d
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
import networkx as nx
from multiprocessing import Pool
import time
import copy
import superpose3d
import subprocess
import shlex
import shutil
import random
from subprocess import run, DEVNULL


np.set_printoptions(threshold=np.inf,suppress=True,precision=2)

class NNPred:
    BBProb=None
    CAProb=None
    CAProb_clusted=None
    AAPred=None
    AAProb=None

class Sequence:
    def __init__(self,fasta_name,sequence):
        self.fasta_name = fasta_name
        self.sequence = sequence
        self.chain_dict = {}
        self.seq_matched_traces = []
        self.trace_matched_seqs = []
        self.trace_scores = []
        self.connect_ign=[]
        self.AF2_raw_structs=None
        self.AF2_struct=None
        self.chain_cand_mat=None


class Chain:
    def __init__(self, chain_id, sequence):
        self.sequence=sequence
        self.chain_id = chain_id
        self.trace_list=[]
        self.result = [-1 for _ in sequence]
        self.highConfResult = [-1 for _ in sequence]
        self.seq_matched_traces=[]
        self.trace_matched_seqs=[]


class GlobalParam:
    def __init__(self):
        self.seq_cand_AA_mat=None
        self.neigh_mat=None
        self.fastas=None
        self.CA_cands=None
        self.neighbors2to6=None
        self.chain_cand_mat=None
        self.matched_chains=None
        self.trace_list=None
        self.CAProb=None


    def update(self,AA_score,pair_scores,fastas=None,CA_cands=None,neighbors2to6=None,chain_cand_mat=None,matched_chains=None,trace_list=None,CAProb=None):
        self.seq_cand_AA_mat=AA_score
        self.neigh_mat=pair_scores
        self.fastas=fastas
        self.CA_cands=CA_cands
        self.neighbors2to6=neighbors2to6
        self.chain_cand_mat=chain_cand_mat
        self.matched_chains=matched_chains
        self.trace_list=trace_list
        self.CAProb=CAProb

globalParam = GlobalParam()


def pathWalking(cand,n_hop):
    global globalParam
    traces=[[cand]]
    scores=[1]
    results=[]
    for n in range(n_hop):
        tmp_traces=[]
        tmp_scores=[]
        for i, trace in enumerate(traces):
            cand = trace[-1]
            neigh_list = list(set(globalParam.neighbors2to6[cand])-set(trace))
            for neigh in neigh_list:
                tmp_traces.append(trace+[neigh])
                tmp_scores.append(scores[i]*max(globalParam.neigh_mat[cand, neigh],0.1))
        
        if tmp_traces:
            result = np.zeros([globalParam.neigh_mat.shape[0]])
            for i, trace in enumerate(tmp_traces):
                result[trace[-1]] = max(result[trace[-1]], tmp_scores[i])
            results.append(result)
            traces=tmp_traces
            scores=tmp_scores
        else:
            break
    traces.clear()
    scores.clear()
    return results


def calc_score(traces,chain_ix,this_seq):
    global globalParam
    result=[]
    for trace in traces:
        rmsd_scores=[]
        for mc in globalParam.matched_chains:
            this_coords=[]
            for p in mc[0]:
                this_coords.append(globalParam.CA_cands[trace[p]])
            rmsd_scores.append(superpose3d.Superpose3D(this_coords,mc[1])[0][0]*len(mc[0])/len(trace))
        if rmsd_scores:
            rmsd_score = min(max(np.mean(rmsd_scores)-1,0)/2,3)
        else:
            rmsd_score = 0

        neigh_score=globalParam.neigh_mat[trace[:-1],trace[1:]].mean()
        AA_score=globalParam.chain_cand_mat[chain_ix, this_seq, trace].mean()
        result.append(neigh_score+AA_score-rmsd_score)
    return result


def localSeqStructAlign(fasta_ix,fasta_name,sub_seq):
    global globalParam
    AF2_split = globalParam.fastas[fasta_name].AF2_struct[sub_seq]
    score_list=[]
    for trace in globalParam.trace_list:
        AA_score =globalParam.seq_cand_AA_mat[fasta_ix,sub_seq,trace].mean()
        nei_score =globalParam.neigh_mat[trace[:-1],trace[1:]].mean()
        this_coords = globalParam.CA_cands[trace]
        af2_rmsd = superpose3d.Superpose3D(this_coords,AF2_split)[0][0]
        score_list.append([AA_score,nei_score,af2_rmsd])
    return score_list


def registerScoring(fasta_ix,fasta_name,seq_ix,radius):
    global globalParam
    this_seq=range(seq_ix-radius,seq_ix+radius+1)
    this_fasta=globalParam.fastas[fasta_name]
    AF2_split = this_fasta.AF2_struct[this_seq]
    chain_num = len(this_fasta.chain_dict)
    chain_list = list(this_fasta.chain_dict.keys())
    
    item_list=[]
    score_list=[]
    
    cand_set=np.where(globalParam.seq_cand_AA_mat[fasta_ix,seq_ix] > globalParam.seq_cand_AA_mat[fasta_ix,seq_ix].max()*0.85)[0]
    for cand in cand_set:
        trace=[cand]
        for i in range(radius):
            max_score=-1
            max_nei=-1
            mean_score = globalParam.seq_cand_AA_mat[fasta_ix, seq_ix+1+i].mean()
            for nei in set(globalParam.neighbors2to6[trace[-1]]) - set(trace):
                score = globalParam.seq_cand_AA_mat[fasta_ix, seq_ix+1+i, nei]
                if score > max_score:
                    max_nei = nei
                    max_score = score
            if max_score > mean_score:
                trace=trace+[max_nei]
            else:
                trace=[]
                break

            max_score=-1
            max_nei=-1
            mean_score = globalParam.seq_cand_AA_mat[fasta_ix, seq_ix-1-i].mean()
            for nei in set(globalParam.neighbors2to6[trace[0]]) - set(trace):
                score = globalParam.seq_cand_AA_mat[fasta_ix, seq_ix-1-i, nei]
                if score > max_score:
                    max_nei = nei
                    max_score = score
            if max_score > mean_score:
                trace=[max_nei]+trace
            else:
                trace=[]
                break
        if trace:
            this_coords = globalParam.CA_cands[trace]
            rmsd, R,T,_ = superpose3d.Superpose3D(this_coords,AF2_split)
            # if rmsd[0]<8:
            new_AF2 = np.dot(this_fasta.AF2_struct, R.T)+T
            trans_AF2 = np.round(new_AF2).astype(int)
            trans_AF2 = trans_AF2[np.where(np.sum(trans_AF2>=0,axis=1)==3)]
            trans_AF2 = trans_AF2[np.where(np.sum(trans_AF2 < globalParam.CAProb.shape, axis=1) == 3)]
            CA_prob_sum = np.sum(globalParam.CAProb[trans_AF2[:, 0], trans_AF2[:, 1], trans_AF2[:, 2]])
            item_list.append([trace,new_AF2[this_seq]])
            score_list.append(CA_prob_sum)
    
    

    results=[]
    if score_list:
        sort_ix = np.argsort(score_list)[::-1]
        for ix in sort_ix:
            trace,new_AF2 = item_list[ix]
            score = score_list[ix]
            this_coords = globalParam.CA_cands[trace]
            if len(results) < 3*chain_num:
                val=True
                for chain in results:
                    if np.sqrt(np.sum((chain[3]-this_coords)**2, axis=1)).mean()<8:
                        val=False
                        break
                if val:
                    results.append([score,trace,this_seq,new_AF2])
            else:
                break
    
    return results


def run_pulchra(dir,pulchar_path,pdbfile,name):
    cspath = os.path.join(dir,f'chain_split/{name}')
    if os.path.exists(cspath):
        shutil.rmtree(cspath)
    os.makedirs(cspath)

    if not os.path.exists(pdbfile):
        print('ca model', pdbfile,'not exsist!')
        return None

    prefix = pdbfile.split('/')[-1].split('.')[0]
    lastcid = ''
    lastrid = ''
    newlines = []
    if not os.path.exists(cspath):
        os.makedirs(cspath)

    with open(pdbfile, 'r') as aafile:
        lines = aafile.readlines()
        for l in lines:
            #print(l)
            if not l.startswith('ATOM'):
                #newlines.append(l)
                continue
            cid = l[21]
            rid = int(l[22:26])
            if lastcid == '':
                lastcid = cid

            if lastrid == '':
                lastrid = rid
            
            if lastcid != cid or rid-lastrid not in [0,1]:
                if len(newlines)>3:
                    rid_name=f'{lastrid//1000%10}{lastrid//100%10}{lastrid//10%10}{lastrid%10}'
                    with open(os.path.join(cspath,f'{prefix}_{lastcid}_{rid_name}.pdb'), 'w') as newfile:
                        for nl in newlines:
                            newfile.writelines(nl)
                newlines=[]
                lastcid = cid
            newlines.append(l)
            lastrid = rid

        
        if len(newlines)>3:
            rid_name=f'{lastrid//1000%10}{lastrid//100%10}{lastrid//10%10}{lastrid%10}'
            with open(os.path.join(cspath,f'{prefix}_{lastcid}_{rid_name}.pdb'), 'w') as newfile:
                    for nl in newlines:
                        newfile.writelines(nl)


    pulchar_path = os.path.abspath(pulchar_path)
    process_list = []
    n_job_per_node = 30
    filelist = os.listdir(cspath)
    for f in filelist:
        prefix = f.split('.')[0]
        
        if 'rebuilt' not in f and 'pdb' in f and not os.path.exists(os.path.join(cspath , prefix + '.rebuilt.pdb')):
            command = pulchar_path + ' {} -c '.format(f)
            args = shlex.split(command)
            # print(command)
            # print(path+p)
            with open(os.path.join(cspath,'{}.log'.format(prefix)), 'w') as log:
                if len(process_list) < n_job_per_node:
                    process_list.append(subprocess.Popen(args, cwd = cspath, stdout=log))
                else:
                    have_finished = False
                    while True:
                        for i in range(len(process_list)):
                            if process_list[i].poll() is not None:
                                process_list[i] = subprocess.Popen(args, cwd = cspath, stdout=log)
                                have_finished = True
                                break
                        if have_finished:
                            break
                        time.sleep(0.5)
            # print(f)
    for p in process_list:
        p.wait()
    
    files = os.listdir(cspath)
        
    file_list = []
    for f in files:
        if '.rebuilt' in f:
            file_list.append(f)
    file_list.sort()
    
    aid = 1
    all_atom_model=pdbfile.split('.pdb')[0] + '_all_atom_model.pdb'
    with open(all_atom_model, 'w') as acf:
        for f in file_list:
            cid = f.split('_ca_model')[-1].split('_')[1]
            with open(os.path.join(cspath,f), 'r') as pf:
                lines = pf.readlines()
                for l in lines:
                    if l.startswith('ATOM') and 'nan' not in l:
                        if len(l)<70:
                            acf.write(l[:4] + str(aid).rjust(7, ' ') + l[11:21] + cid + l[22:54])
                            acf.write(f'  1.00  0.00           {l[13]}\n')
                        else:
                            acf.write(l[:4] + str(aid).rjust(7, ' ') + l[11:21] + cid + l[22:])
                        aid+=1
    return all_atom_model


def get_seq(inp):
    seq_obj,chain_strs = inp
    un_exist=False
    
    return seq_obj,chain_strs,un_exist

def interpolate_missing_cas(ca_array):
    """
    补齐缺失的 CA 原子坐标
    :param ca_coordinates: 包含 CA 原子坐标的列表，其中 None 表示缺失的坐标
    :return: 补齐后的 CA 原子坐标列表
    """

    # 找到缺失的坐标索引
    missing_indices = np.where(np.isnan(ca_array[:, 0]))[0]

    for index in missing_indices:
        # 找到前一个有效的 CA 原子坐标
        prev_index = index - 1
        while prev_index >= 0 and np.isnan(ca_array[prev_index, 0]):
            prev_index -= 1

        # 找到后一个有效的 CA 原子坐标
        next_index = index + 1
        while next_index < len(ca_array) and np.isnan(ca_array[next_index, 0]):
            next_index += 1

        # 进行线性插值
        if prev_index >= 0 and next_index < len(ca_array):
            ca_array[index] = (ca_array[prev_index] + ca_array[next_index]) / 2 + np.random.randn(3)
        elif prev_index >= 0:
            ca_array[index] = ca_array[prev_index] + np.random.randn(3)
        else:
            ca_array[index] = ca_array[next_index] + np.random.randn(3)

    return ca_array


class CryoProcesser:
    def __init__(self,dynamic_config):
        # init
        if dynamic_config.protocol =='temp_free':
            self.method_name='seq'
        elif 'seq_free' in dynamic_config.protocol:
            self.method_name='noseq'
        try:
            self.emid = dynamic_config.map_path.split('/')[-1].split('emd_')[-1].split('.')[0]
        except:
            self.emid='unknownEM'
        try:
            self.pdbid = dynamic_config.fasta_path.split('/')[-1].split('.fasta')[0]
        except:
            self.pdbid='unknownPDB'
        self.resol = dynamic_config.resolution
        print(f'starting... emid: {self.emid}, pdbid: {self.pdbid}')
        self.dynamic_config = dynamic_config
        torch.manual_seed(dynamic_config.seed)

        # processEM
        self.normEM=None
        
        # clustering
        self.neighbors2to6 = []
        self.neighbors0to6 = []
        self.neighbors2to7 = []
        self.neighbors0to7 = []

        # fragModeling
        self.fragModel=[]

        # checkSeq
        self.ResNum=0
        self.max_seq_len=0
        self.fasta_list=[]
        self.fastas = {}
        self.chain_id_list=[]

        # gtGT
        self.pdbParser=PDBParser(PERMISSIVE=1)

        # time
        self.time_cost={}

        self.cluster_eps = dynamic_config.cluster_eps
        self.cluster_min_points = dynamic_config.cluster_min_points
        self.nms_radius=dynamic_config.nms_radius
    

    def get_spatial_feature(self,BB_model,CA_model,AA_model):
        if 'seq_free' not in self.dynamic_config.protocol:
            checkSeqRes=self.checkSeq()
            if checkSeqRes != 'success':
                return checkSeqRes
            
        self.nnProcess(BB_model,CA_model,AA_model)

    def mapping(self):
        start_time=time.time()
        print('fragModeling...')
        X, C, S = self.fragModeling()
        if 'seq_free' in self.dynamic_config.protocol:
            return X, C, S
        
        self.time_cost['fragModeling']=time.time()-start_time
        print('fragModeling finished in {} seconds'.format(round(self.time_cost['fragModeling'])))

        run_result=self.seqMapAligning()
        if run_result != 'success':
            print(run_result)
            return
        
        start_time=time.time()
        print('initialModeling...')
        self.initialModeling()
        self.time_cost['initialModeling']=time.time()-start_time
        print('initialModeling finished in {} seconds'.format(round(self.time_cost['initialModeling'])))

        start_time=time.time()
        print('gapThreading...')
        X, C, S = self.gapThreading()
        return X, C, S

    def get_seq(self):
        C, S = [], []
        chain_counts = 1
        for fasta_ix, fasta_name in enumerate(self.fastas):
            this_fasta= self.fastas[fasta_name]
            for chain_id in this_fasta.chain_dict:
                res = np.array(this_fasta.chain_dict[chain_id].result)
                length = len(res)
                C += [chain_counts]*length
                S += this_fasta.sequence
                atoms = self.CA_cands[res]
                atoms[res==-1] = np.nan
                chain_counts += 1
        C = torch.LongTensor(C)
        return C, S
            

    def nnProcess(self,BB_model,CA_model,AA_model):
        start_time=time.time()
        print('preprocessEM...')
        self.preprocessEM()
        self.time_cost['preprocessEM']=time.time()-start_time
        print('preprocessEM finished in {} seconds'.format(round(self.time_cost['preprocessEM'])))

        start_time=time.time()
        print('nnPred...')
        self.nnPred(BB_model,CA_model,AA_model)
        self.time_cost['nnPred']=time.time()-start_time
        print('nnPred finished in {} seconds'.format(round(self.time_cost['nnPred'])))

        start_time=time.time()
        print('clustering...')
        self.clustering()
        self.time_cost['clustering']=time.time()-start_time
        print('clustering finished in {} seconds'.format(round(self.time_cost['clustering'])))


    def fragModeling(self):
        cand_graph=nx.Graph()
        edge_list=[]
        for cand in range(self.CA_cands.shape[0]):
            for neigh in self.neighbors2to6[cand]:
                if neigh>cand:
                    cand_graph.add_edge(cand,neigh)
                    edge_list.append([self.neigh_mat[cand,neigh],cand,neigh])
        
        edge_list = np.array(edge_list)
        edge_list = edge_list[np.argsort(edge_list[:, 0], axis=0)]
        new_edge_list=[]
        for edge in edge_list:
            cand = round(edge[1])
            neigh = round(edge[2])
            if cand_graph.degree(cand) > 2 and cand_graph.degree(neigh) > 2:
                cand_graph.remove_edge(cand,neigh)
            else:
                new_edge_list.append([self.neigh_mat[cand,neigh],cand,neigh])

        edge_list = np.array(new_edge_list)
        edge_list = edge_list[np.argsort(edge_list[:, 0], axis=0)]
        for edge in edge_list:
            cand = round(edge[1])
            neigh = round(edge[2])
            if cand_graph.degree(cand) > 2 or cand_graph.degree(neigh) > 2:
                cand_graph.remove_edge(cand,neigh)

        fragments=[]
        tmp_grpah = cand_graph.copy()
        for node in cand_graph.nodes:
            if  tmp_grpah.degree(node) == 1:
                next = list(tmp_grpah[node])[0]
                frag=[node,next]
                tmp_grpah.remove_edge(node, next)
                while (tmp_grpah.degree(next) == 1):
                    neigh = list(tmp_grpah[next])[0]
                    frag.append(neigh)
                    tmp_grpah.remove_edge(next, neigh)
                    next = neigh
                fragments.append(frag)

        while(len(tmp_grpah.edges()) > 0):
            edge_scores = []
            for node, neigh in tmp_grpah.edges():
                edge_scores.append(
                    [self.neigh_mat[node, neigh], node, neigh])

            edge_scores = np.array(edge_scores)
            min_edge = edge_scores[np.argmin(edge_scores[:, 0])]
            node = round(min_edge[1])
            tmp_grpah.remove_edge(node,round(min_edge[2]))
            if tmp_grpah.degree(node)==1:
                next = list(tmp_grpah[node])[0]
                frag = [node, next]
                tmp_grpah.remove_edge(node, next)
                while (tmp_grpah.degree(next) == 1):
                    neigh = list(tmp_grpah[next])[0]
                    frag.append(neigh)
                    tmp_grpah.remove_edge(next, neigh)
                    next = neigh
                fragments.append(frag)

        if len(fragments)> min(62,self.CA_cands.shape[0]//self.dynamic_config.frags_len+1):
            tmp_fragments=copy.deepcopy(fragments)
            
            while len(tmp_fragments) > min(62,self.CA_cands.shape[0]//self.dynamic_config.frags_len+1):
                disMap = np.full((2*len(tmp_fragments),2*len(tmp_fragments)),10000)
                for i,frag1 in enumerate(tmp_fragments):
                    for j,frag2 in enumerate(tmp_fragments):
                        if i != j:
                            disMap[2*i,2*j]=self.cand_self_dis[frag1[0],frag2[0]]
                            disMap[2*i+1,2*j]=self.cand_self_dis[frag1[-1],frag2[0]]
                            disMap[2*i,2*j+1]=self.cand_self_dis[frag1[0],frag2[-1]]
                            disMap[2*i+1,2*j+1]=self.cand_self_dis[frag1[-1],frag2[-1]]
                best_ix=np.unravel_index(disMap.argmin(),disMap.shape)
                best_i,best_j = best_ix[0]//2,best_ix[1]//2
                left_trace = copy.deepcopy(tmp_fragments[best_i] if best_ix[0]%2==1 else tmp_fragments[best_i][::-1])
                right_trace = copy.deepcopy(tmp_fragments[best_j] if best_ix[1]%2==0 else tmp_fragments[best_j][::-1])

                new_frag=left_trace+right_trace
                if best_i > best_j:
                    del tmp_fragments[best_i]
                    del tmp_fragments[best_j]
                else:
                    del tmp_fragments[best_j]
                    del tmp_fragments[best_i]
                tmp_fragments.append(new_frag)
            fragments = tmp_fragments
         

        print('seq_free trace_num:',len(fragments))
        sort_ind = np.argsort([len(frag) for frag in fragments])[::-1]
        X, S = [], []
        C = torch.cat([torch.LongTensor([i+1]*len(fragments[ind])) for i, ind in enumerate(sort_ind)], dim=0) 
        for ind in sort_ind:
            CA_indexes = fragments[ind]
            X.append(self.CA_cands[CA_indexes])
            S += [utils.AA_abb_T[self.CA_cands_AA[i]] for i in CA_indexes]
        X = torch.from_numpy(np.concatenate(X, axis=0)) + torch.Tensor(self.offset)[None,:]
        
        return X, C, ''.join(S)



    def seqMapAligning(self):
        self.prepareSeq4Align()
        start_time=time.time()
        print('seqStructureAlign...')
        if not self.seqStructureAlign():
            return 'seqStructureAlign error! this case is too hard!'
        self.time_cost['seqStructureAlign']=time.time()-start_time
        print('seqStructureAlign finished in {} seconds'.format(round(self.time_cost['seqStructureAlign'])))

        return 'success'


    def initialModeling(self):
        for fasta_ix in range(len(self.alignedFrags)):
            self.fastas[self.fasta_list[fasta_ix]].seq_matched_traces=[]
            self.fastas[self.fasta_list[fasta_ix]].trace_matched_seqs=[]
            self.fastas[self.fasta_list[fasta_ix]].trace_scores=[]
            for fragment in self.alignedFrags[fasta_ix]:
                self.fastas[self.fasta_list[fasta_ix]].seq_matched_traces.append(fragment[0])
                self.fastas[self.fasta_list[fasta_ix]].trace_matched_seqs.append(fragment[1])
                AA_score = self.seq_cand_AA_mat[fasta_ix,fragment[1],fragment[0]]
                neigh_score=self.neigh_mat[fragment[0][:-1],fragment[0][1:]]
                self.fastas[self.fasta_list[fasta_ix]].trace_scores.append((AA_score[1:]+AA_score[:-1])*neigh_score)

        self.used_cands=set()
        matched_len_thrh=7
        unmatched_rmsd_thrh=4
        for fasta_ix, fasta_name in enumerate(self.fastas):
            this_fasta = self.fastas[fasta_name]
            score_lists=[]
            matched_traces=[]
            unused_traces=set([_ for _ in range(len(this_fasta.trace_matched_seqs))])
            for seq_ix in range(len(this_fasta.sequence)):
                matched_trace=[]
                score_list=[]
                for s,seqs in enumerate(this_fasta.trace_matched_seqs):
                    if seq_ix in seqs:
                        i = seq_ix-seqs[0]
                        partion= i/len(seqs)
                        score_list.append(np.sum(this_fasta.trace_scores[s])+2*partion*(1-partion))
                        matched_trace.append(s)

                matched_traces.append(np.array(matched_trace)[np.argsort(score_list)[::-1]])
                score_lists.append(np.sum(score_list))

            if not matched_traces:
                continue
            max_seq_ix = np.argmax(score_lists)

            chain_list=list(this_fasta.chain_dict.keys())

            model={}
            for id in matched_traces[max_seq_ix]:
                if len(model)<len(chain_list):
                    model[chain_list[len(model)]]=[id]
                    unused_traces.discard(id)

            models=[model]
            left_seq=max_seq_ix
            right_seq=max_seq_ix
            while True:
                tmp_models=[]
                for trace_id in copy.deepcopy(unused_traces):
                    seqs = this_fasta.trace_matched_seqs[trace_id]
                    traces = this_fasta.seq_matched_traces[trace_id]
                    if left_seq in seqs:
                        if len(models[0]) < len(chain_list):
                            models[0][chain_list[len(models[0])]]=[trace_id]#confuse, change from models[0][chain_list[len(model)]]=[trace_id]
                            break
                        for model in models:
                            matched_chain_ids=set()
                            for chain_id in model:
                                for ti in model[chain_id]:
                                    chain = this_fasta.trace_matched_seqs[ti]
                                    if len(set(seqs)&set(chain))>matched_len_thrh:#change from if len(set(seqs)&set(chain))>4:
                                        matched_chain_ids.add(chain_id)
                            unmatched_chain_ids = set([_ for _ in chain_list])-matched_chain_ids
                            if not unmatched_chain_ids:
                                tmp_models.append(copy.deepcopy(model))
                            elif self.dynamic_config.protocol == 'temp_flex' or matched_chain_ids:
                                if self.dynamic_config.protocol == 'temp_flex':#######..................
                                    rmsd_mat=np.full((len(matched_chain_ids)+1,len(unmatched_chain_ids)),10000.0)
                                else:
                                    rmsd_mat=np.full((len(matched_chain_ids),len(unmatched_chain_ids)),10000.0)
                                occ_chain_lists=[]
                                for i, chain_i in enumerate(matched_chain_ids):
                                    chain = model[chain_i]
                                    occ_chain_list=[-1 for _ in range(len(this_fasta.sequence))]
                                    for id in chain:
                                        for s, seq_ix in enumerate(this_fasta.trace_matched_seqs[id]):
                                            occ_chain_list[seq_ix]=this_fasta.seq_matched_traces[id][s]
                                    occ_chain_lists.append(occ_chain_list)

                                val_chain_lists=[]
                                for i, chain_i in enumerate(unmatched_chain_ids):
                                    chain = model[chain_i]
                                    val_chain_list=[-1 for _ in range(len(this_fasta.sequence))]
                                    for id in chain:
                                        for s, seq_ix in enumerate(this_fasta.trace_matched_seqs[id]):
                                            val_chain_list[seq_ix]=this_fasta.seq_matched_traces[id][s]
                                    for s, seq_ix in enumerate(seqs):
                                        val_chain_list[seq_ix]=traces[s]
                                    val_chain_lists.append(val_chain_list)
                                
                                
                                for j, chain_j in enumerate(unmatched_chain_ids):
                                    for i, chain_i in enumerate(matched_chain_ids):
                                        val_coords=[]
                                        occ_coords=[]
                                        for s in range(len(this_fasta.sequence)):
                                            if occ_chain_lists[i][s]!=-1 and val_chain_lists[j][s]!=-1:
                                                occ_coords.append(self.CA_cands[occ_chain_lists[i][s]])
                                                val_coords.append(self.CA_cands[val_chain_lists[j][s]])
                                        rmsd_mat[i,j] = superpose3d.Superpose3D(val_coords,occ_coords)[0][0]
                                    if self.dynamic_config.protocol == 'temp_flex':
                                        occ_coords=[]
                                        val_coords=[]
                                        for s in range(len(this_fasta.sequence)):
                                            if val_chain_lists[j][s]!=-1:
                                                occ_coords.append(this_fasta.AF2_struct[s])
                                                val_coords.append(self.CA_cands[val_chain_lists[j][s]])
                                        rmsd_mat[-1,j] = superpose3d.Superpose3D(val_coords,occ_coords)[0][0]

                                min_i,min_j = np.unravel_index(np.argmin(rmsd_mat),rmsd_mat.shape)
                                if rmsd_mat[min_i,min_j] < unmatched_rmsd_thrh:#new condition
                                    tmp_model=copy.deepcopy(model)
                                    tmp_model[list(unmatched_chain_ids)[min_j]] = [trace_id] + tmp_model[list(unmatched_chain_ids)[min_j]]
                                    tmp_models.append(tmp_model)
                                else:
                                    tmp_models.append(copy.deepcopy(model))
                            else:
                                for chain_id in unmatched_chain_ids:
                                    chain = model[chain_id]
                                    tmp_model=copy.deepcopy(model)
                                    tmp_model[chain_id] = [trace_id] + tmp_model[chain_id]
                                    tmp_models.append(tmp_model)
                        unused_traces.discard(trace_id)
                        break
                        

                    if right_seq in seqs:
                        if len(models[0]) < len(chain_list):
                            models[0][chain_list[len(models[0])]]=[trace_id]#confuse, change from models[0][chain_list[len(model)]]=[trace_id]
                            break
                        for model in models:
                            matched_chain_ids=set()
                            for chain_id in model:
                                for ti in model[chain_id]:
                                    chain = this_fasta.trace_matched_seqs[ti]
                                    if len(set(seqs)&set(chain))>matched_len_thrh:#change from if len(set(seqs)&set(chain))>4:
                                        matched_chain_ids.add(chain_id)
                            unmatched_chain_ids = set([_ for _ in chain_list])-matched_chain_ids
                            if not unmatched_chain_ids:
                                tmp_models.append(copy.deepcopy(model))
                            elif self.dynamic_config.protocol == 'temp_flex' or matched_chain_ids:
                                if self.dynamic_config.protocol == 'temp_flex':
                                    rmsd_mat=np.full((len(matched_chain_ids)+1,len(unmatched_chain_ids)),10000.0)
                                else:
                                    rmsd_mat=np.full((len(matched_chain_ids),len(unmatched_chain_ids)),10000.0)
                                occ_chain_lists=[]
                                for i, chain_i in enumerate(matched_chain_ids):
                                    chain = model[chain_i]
                                    occ_chain_list=[-1 for _ in range(len(this_fasta.sequence))]
                                    for id in chain:
                                        for s, seq_ix in enumerate(this_fasta.trace_matched_seqs[id]):
                                            occ_chain_list[seq_ix]=this_fasta.seq_matched_traces[id][s]
                                    occ_chain_lists.append(occ_chain_list)

                                val_chain_lists=[]
                                for i, chain_i in enumerate(unmatched_chain_ids):
                                    chain = model[chain_i]
                                    val_chain_list=[-1 for _ in range(len(this_fasta.sequence))]
                                    for id in chain:
                                        for s, seq_ix in enumerate(this_fasta.trace_matched_seqs[id]):
                                            val_chain_list[seq_ix]=this_fasta.seq_matched_traces[id][s]
                                    for s, seq_ix in enumerate(seqs):
                                        val_chain_list[seq_ix]=traces[s]
                                    val_chain_lists.append(val_chain_list)

                                for j, chain_j in enumerate(unmatched_chain_ids):
                                    for i, chain_i in enumerate(matched_chain_ids):
                                        val_coords=[]
                                        occ_coords=[]
                                        for s in range(len(this_fasta.sequence)):
                                            if occ_chain_lists[i][s]!=-1 and val_chain_lists[j][s]!=-1:
                                                occ_coords.append(self.CA_cands[occ_chain_lists[i][s]])
                                                val_coords.append(self.CA_cands[val_chain_lists[j][s]])
                                        rmsd_mat[i,j] = superpose3d.Superpose3D(val_coords,occ_coords)[0][0]
                                    if self.dynamic_config.protocol == 'temp_flex':
                                        occ_coords=[]
                                        val_coords=[]
                                        for s in range(len(this_fasta.sequence)):
                                            if val_chain_lists[j][s]!=-1:
                                                occ_coords.append(this_fasta.AF2_struct[s])
                                                val_coords.append(self.CA_cands[val_chain_lists[j][s]])
                                        rmsd_mat[-1,j] = superpose3d.Superpose3D(val_coords,occ_coords)[0][0]

                                min_i,min_j = np.unravel_index(np.argmin(rmsd_mat),rmsd_mat.shape)
                                if rmsd_mat[min_i,min_j] < unmatched_rmsd_thrh:#new condition
                                    tmp_model=copy.deepcopy(model)
                                    tmp_model[list(unmatched_chain_ids)[min_j]] =  tmp_model[list(unmatched_chain_ids)[min_j]]+[trace_id]
                                    tmp_models.append(tmp_model)
                                else:
                                    tmp_models.append(copy.deepcopy(model))
                            else:
                                for chain_id in unmatched_chain_ids:
                                    chain = model[chain_id]
                                    tmp_model=copy.deepcopy(model)
                                    tmp_model[chain_id] = tmp_model[chain_id]+[trace_id]
                                    tmp_models.append(tmp_model)

                        unused_traces.discard(trace_id)
                        break
                if tmp_models:
                    if len(tmp_models)>1000:
                        dis_list=[]
                        for model in tmp_models:
                            dis=[]
                            for chain_id in model:
                                for i,ti in enumerate(model[chain_id][:-1]):
                                    cand1=this_fasta.seq_matched_traces[ti][-1]
                                    cand2=this_fasta.seq_matched_traces[model[chain_id][i+1]][0]
                                    seq1=this_fasta.trace_matched_seqs[ti][-1]
                                    seq2=this_fasta.trace_matched_seqs[model[chain_id][i+1]][0]
                                    sp_dis= self.cand_self_dis[cand1,cand2]
                                    seq_dis=abs(seq2-seq1)
                                    dis.append(np.sqrt(seq_dis)+sp_dis+sp_dis/(seq_dis+1))
                            dis_list.append(np.mean(dis))
                        sort_ix=np.argsort(dis_list)
                        models=[]
                        for i in range(10):
                            models.append(tmp_models[sort_ix[i]])
                    else:
                        models=tmp_models
                elif left_seq>-1 or right_seq<len(this_fasta.sequence):
                    if left_seq>-1:
                        left_seq-=1
                    if right_seq<len(this_fasta.sequence):
                        right_seq+=1
                else:
                    break

            dis_list=[]
            for model in models:
                dis=[]
                for chain_id in model:
                    for i,ti in enumerate(model[chain_id][:-1]):
                        cand1=this_fasta.seq_matched_traces[ti][-1]
                        cand2=this_fasta.seq_matched_traces[model[chain_id][i+1]][0]
                        seq1=this_fasta.trace_matched_seqs[ti][-1]
                        seq2=this_fasta.trace_matched_seqs[model[chain_id][i+1]][0]
                        sp_dis= self.cand_self_dis[cand1,cand2]
                        seq_dis=abs(seq2-seq1)
                        dis.append(np.sqrt(seq_dis)+sp_dis+sp_dis/(seq_dis+1))
                dis_list.append(np.mean(dis))
            min_ix=np.argmin(dis_list)
            model = models[min_ix]

            for j,chain_id in enumerate(model):
                chain = model[chain_id]
                score_list=[]
                for ix in chain:
                    score_list.append(np.sum(this_fasta.trace_scores[ix]))
                arg_score_ix = np.argsort(score_list)
                for i in range(len(chain)):
                    ix = chain[arg_score_ix[i]]
                    for c, cand in enumerate(this_fasta.seq_matched_traces[ix][3:-3]):
                        p = this_fasta.trace_matched_seqs[ix][3:-3][c]
                        this_fasta.chain_dict[chain_id].result[p] = cand
                for i in range(len(chain)):
                    for cand in this_fasta.chain_dict[chain_id].result:
                        if cand !=-1:
                            self.used_cands.add(cand)


    def gapThreading(self):
        atom_ix=0
        for fasta_ix, fasta_name in enumerate(self.fastas):
            this_fasta = self.fastas[fasta_name]

            chain_cand_score=np.zeros([len(this_fasta.chain_dict),self.seq_cand_AA_mat.shape[1],self.seq_cand_AA_mat.shape[2]])
            chain_list=list(this_fasta.chain_dict.keys())
            for i in range(chain_cand_score.shape[0]):
                chain_id = chain_list[i]
                
                this_fasta.chain_dict[chain_id].highConfResult = copy.copy(this_fasta.chain_dict[chain_id].result)
                for c in range(chain_cand_score.shape[2]):
                    if c not in self.used_cands:
                        chain_cand_score[i,:,c]=self.seq_cand_AA_mat[fasta_ix,:,c]
                for p,cand in enumerate(this_fasta.chain_dict[chain_id].result):
                    if cand!=-1:
                        chain_cand_score[i,p,:]=0
                        chain_cand_score[:,:,cand]=0#new
                        chain_cand_score[i,p,cand]=1

            chain_cand_mat=chain_cand_score.copy()
            for i in range(self.dynamic_config.n_hop):
                chain_cand_mat+=np.pad(chain_cand_score[:,:-i-1,:],[(0,0),(i+1,0),(0,0)],constant_values=(0,0))@self.n_hop_mat[i].T+np.pad(chain_cand_score[:,i+1:,:],[(0,0),(0,i+1),(0,0)],constant_values=(0,0))@self.n_hop_mat[i].T
            for c in self.used_cands:
                chain_cand_mat[:,:,c]=0#new
            this_fasta.chain_cand_mat=chain_cand_mat

            start_ends=[]
            for i, chain_id in enumerate(this_fasta.chain_dict):
                start_end=[]
                init_model=this_fasta.chain_dict[chain_id].result
                pair=[]
                for t, cand in enumerate(init_model):
                    if cand == -1:
                        if not pair:
                            pair=[t-1]
                    else:
                        if pair:
                            pair.append(t)
                            start_end.append([pair[0],pair[1]])
                            start_ends.append([i,set(range(pair[0]+1,pair[1])),pair[0],pair[1]])
                            pair=[]
                if pair:
                    pair.append(len(init_model))
                    start_end.append([pair[0],pair[1]])
                    start_ends.append([i,set(range(pair[0]+1,pair[1])),pair[0],pair[1]])

            unmat_len_list=[]
            for start_end1 in start_ends:
                unmat_len=0
                set1=start_end1[1]
                for start_end2 in start_ends:
                    unmat_len+=len(set1&start_end2[1])
                unmat_len_list.append(unmat_len)
            sort_ix = np.argsort(unmat_len_list)
            for e,ix in enumerate(sort_ix):
                print(e,'/',len(unmat_len_list),start_ends[ix][2],start_ends[ix][3])

                self.fillGap(fasta_ix,start_ends[ix])

        
        cand_occ={}
        centroids={}
        for fasta_ix, fasta_name in enumerate(self.fastas):
            this_fasta= self.fastas[fasta_name]
            for chain_id in this_fasta.chain_dict:
                coord_list=[]
                for seq_id, cand in enumerate(this_fasta.chain_dict[chain_id].highConfResult):
                    if cand!=-1:
                        coord_list.append(self.CA_cands[cand])
                if coord_list:
                    centroids[(fasta_name,chain_id)] = np.array(coord_list).mean(axis=0)
                for seq_id, cand in enumerate(this_fasta.chain_dict[chain_id].result):
                    if cand!=-1:
                        if cand not in cand_occ:
                            cand_occ[cand]=[]
                        cand_occ[cand].append([fasta_name,chain_id,seq_id])
        
        for cand in cand_occ:
            min_dis=10000
            for fasta_name,chain_id,seq_id in cand_occ[cand]:
                dis2=np.sum((centroids[(fasta_name,chain_id)]-self.CA_cands[cand])**2)
                min_dis=min(min_dis, dis2)
            
            for fasta_name,chain_id,seq_id in cand_occ[cand]:
                this_fasta= self.fastas[fasta_name]
                dis2=np.sum((centroids[(fasta_name,chain_id)]-self.CA_cands[cand])**2)
                if dis2>min_dis+1:
                    seq_len=len(this_fasta.sequence)
                    for s in range(max(0,seq_id-2),min(seq_len,seq_id+3)):
                        if this_fasta.chain_dict[chain_id].highConfResult[s] != -1:
                            continue
                        this_fasta.chain_dict[chain_id].result[s]=-1

        cand_set=set()

        X, C, S = [], [], []
        chain_counts=1
        for fasta_ix, fasta_name in enumerate(self.fastas):
            this_fasta= self.fastas[fasta_name]
            for chain_id in this_fasta.chain_dict:
                res = np.array(this_fasta.chain_dict[chain_id].result)
                mask = (res!=-1)
                res = res[mask]
                length = len(res)
                C += [chain_counts]*length
                S += [this_fasta.sequence[i] for i in range(len(mask)) if mask[i]]
                atoms = self.CA_cands[res]
                atoms[res==-1] = 0
                X.append(atoms)
            #    X.append(interpolate_missing_cas(atoms))
                chain_counts += 1
        X = torch.from_numpy(np.concatenate(X, axis=0))
        C = torch.LongTensor(C)
        return X, C, S
    

    
    def phenix_refine(self,all_atom_model):
        phenix_param=os.path.abspath(self.dynamic_config.phenix_param)
        phenix_act=os.path.abspath(self.dynamic_config.phenix_act)
        output_dir=os.path.abspath(self.dynamic_config.output_dir)
        map_path=os.path.abspath(self.dynamic_config.map_path)
        all_atom_model=os.path.abspath(all_atom_model)
        cmd=f'bash phenix.sh {phenix_act} {output_dir} "phenix.real_space_refine {all_atom_model} {map_path} {phenix_param} resolution={self.resol}"'
        run([cmd], stdout=DEVNULL, stderr=DEVNULL, shell=True)


    def time_record(self):
        with open(self.time_log,'w') as w:
            w.write('step,time\n')
            for key in self.time_cost:
                w.write('{},{}\n'.format(key,round(self.time_cost[key])))


    def preprocessEM(self):
        EMmap = mrcfile.open(self.dynamic_config.map_path)
        self.normEM, self.offset = utils.processEMData(EMmap)
        EMmap.close()


    def nnPred(self,BB_model,CA_model,AA_model):
        with torch.no_grad():
            padded_em = np.pad(self.normEM, [(8, 64 - (self.normEM.shape[0]) % 48), (8, 64 - (self.normEM.shape[1]) %
                            48), (8, 64 - (self.normEM.shape[2]) % 48)], 'constant', constant_values=[(0, 0), (0, 0), (0, 0)])
            NNPred.BBProb = np.zeros_like(self.normEM)
            self.CAProb = np.zeros_like(self.normEM)
            self.AAPred = np.zeros_like(self.normEM)
            NNPred.AAProb = np.zeros([20,self.normEM.shape[0], self.normEM.shape[1], self.normEM.shape[2]])

            for i in range(0, self.normEM.shape[0], 48):
                for j in range(0, self.normEM.shape[1], 48):
                    di = min(48, self.normEM.shape[0]-i)
                    dj = min(48, self.normEM.shape[1]-j)

                    sub_em = torch.zeros([self.normEM.shape[2]//48+1,1,64,64,64]).to('cuda')
                    for k in range(0, self.normEM.shape[2], 48):
                        sub_em[k//48,0,:,:,:]=torch.FloatTensor(padded_em[i:i+64, j:j+64, k:k+64]).to('cuda')

                    softm = torch.nn.Softmax(dim=1)
                    BB_output,_,_ = BB_model(sub_em)
                    _,CA_output,_ = CA_model(sub_em)
                    _,_,AA_output = AA_model(sub_em)
                    BB_output = torch.cat((BB_output[:,:1],BB_output[:,2:]),axis=1)
                    score_BB = softm(BB_output)
                    CA_output = torch.cat((CA_output[:,:1],CA_output[:,2:]),axis=1)
                    score_CA = softm(CA_output)
                    score_AA = softm(AA_output[:,1:,:,:,:])
                    scoreMax_AA = torch.max(score_AA, 1)[1]

                    for k in range(0, self.normEM.shape[2], 48):
                        dk = min(48, self.normEM.shape[2]-k)
                        NNPred.BBProb[i:i+di, j:j+dj, k:k+dk] = score_BB[k//48,2, 8:8+di, 8:8+dj, 8:8+dk].cpu().detach().numpy()
                        self.CAProb[i:i+di, j:j+dj, k:k+dk] = score_CA[k//48,2, 8:8+di, 8:8+dj, 8:8+dk].cpu().detach().numpy()
                        self.AAPred[i:i+di, j:j+dj, k:k+dk] = scoreMax_AA[k//48,8:8+di, 8:8+dj, 8:8+dk].cpu().detach().numpy()
                        NNPred.AAProb[:,i:i+di, j:j+dj, k:k+dk] = score_AA[k//48,:, 8:8+di, 8:8+dj, 8:8+dk].cpu().detach().numpy()


    def clustering(self):
        pcd_numpy = np.array(np.where(self.CAProb > self.dynamic_config.CA_score_thrh)).T
        pcd_numpy = pcd_numpy.astype(np.float64)
        pcd_raw = o3d.geometry.PointCloud()

        if not pcd_numpy.flags['C_CONTIGUOUS']:
            pcd_numpy = np.ascontiguousarray(pcd_numpy)
        pcd_raw.points = o3d.utility.Vector3dVector(pcd_numpy)
        labels = np.array(pcd_raw.cluster_dbscan(eps=self.cluster_eps, min_points=self.cluster_min_points))
        pcd_numpy = pcd_numpy.astype(np.int64)

        labels_scores_sum = []
        for label in range(labels.max()+1):
            pcd = pcd_numpy[np.where(labels == label)]
            labels_scores_sum.append(np.sum(NNPred.BBProb[pcd[:,0],pcd[:,1],pcd[:,2]]))

        labels_scores_avg = []
        for label in range(labels.max()+1):
            if labels_scores_sum[label] > np.max(labels_scores_sum)/10:
                pcd = pcd_numpy[np.where(labels == label)]
                labels_scores_avg.append(np.mean(NNPred.BBProb[pcd[:,0],pcd[:,1],pcd[:,2]]))
            else:
                labels_scores_avg.append(0)
        
        
        val_mat=np.zeros_like(labels).astype(bool)
        max_labels_score=np.max(labels_scores_avg)
        for label in range(labels.max()+1):
            if labels_scores_avg[label]>max_labels_score/2:
                val_mat[np.where(labels == label)]=True
        
        clustered_coords=pcd_numpy[np.where(val_mat)]
        NNPred.CAProb_clusted = np.zeros_like(self.CAProb)
        NNPred.CAProb_clusted[clustered_coords[:, 0], clustered_coords[:, 1], clustered_coords[:, 2]] \
            = self.CAProb[clustered_coords[:, 0], clustered_coords[:, 1], clustered_coords[:, 2]]
        pred_list=[]
        indexes = np.where(val_mat)
        for i in range(indexes[0].shape[0]):
            pred_list.append(
                [NNPred.CAProb_clusted[pcd_numpy[indexes[0][i]][0], pcd_numpy[indexes[0][i]][1], pcd_numpy[indexes[0][i]][2]], pcd_numpy[indexes[0][i]][0], pcd_numpy[indexes[0][i]][1], pcd_numpy[indexes[0][i]][2]])
        pred_list = np.array(pred_list)
        pred_list = pred_list[np.argsort(-pred_list[:, 0], axis=0)]
        CA_cands=[]
        clustered_map = np.zeros_like(self.normEM)
        while (pred_list.shape[0] > 0 and pred_list[0][0] >= self.dynamic_config.CA_score_thrh):
            CA_cands.append([int(pred_list[0, 1]), int(pred_list[0, 2]), int(pred_list[0, 3])])
            clustered_map[int(pred_list[0, 1]), int(pred_list[0, 2]), int(pred_list[0, 3])]=1
            delete_list = np.where(
                (pred_list[:, 1] - pred_list[0, 1]) ** 2 + (pred_list[:, 2] - pred_list[0, 2]) ** 2 + (
                        pred_list[:, 3] - pred_list[0, 3]) ** 2 <= self.nms_radius)
            pred_list = np.delete(pred_list, delete_list, 0)

        new_cands=[]
        new_AAs=[]
        for cand in CA_cands:
            try:
                coord=[0,0,0]
                AA_list=[]
                cand=np.array(cand)
                weights=self.CAProb[cand[0]-1:cand[0]+2,cand[1]-1:cand[1]+2,cand[2]-1:cand[2]+2]/np.sum(self.CAProb[cand[0]-1:cand[0]+2,cand[1]-1:cand[1]+2,cand[2]-1:cand[2]+2])
                for di in [-1,0,1]:
                    for dj in [-1,0,1]:
                        for dk in [-1,0,1]:
                            this_coord=cand+[di,dj,dk]
                            coord+=this_coord*weights[di+1,dj+1,dk+1]
                            AA_list.append(NNPred.AAProb[:,this_coord[0],this_coord[1],this_coord[2]]*weights[di+1,dj+1,dk+1])
                new_cands.append(coord)
                new_AAs.append(np.sum(AA_list,axis=0))

            except:
                print('warning!: there are candidate atom in the boundary')

        self.CA_cands = np.array(new_cands)
        self.CA_cands_AAProb = np.array(new_AAs).T
        round_cands= np.round(self.CA_cands).astype(int)
        self.CA_cands_AA = self.AAPred[round_cands[:,0],round_cands[:,1],round_cands[:,2]]

        self.cand_self_dis = utils.calc_dis(self.CA_cands, self.CA_cands)
        for i in range(self.CA_cands.shape[0]):
            self.neighbors2to6.append(np.where((self.cand_self_dis[i] <= 6) * (self.cand_self_dis[i] >= 2))[0])
        for i in range(self.CA_cands.shape[0]):
            self.neighbors0to6.append(np.where(self.cand_self_dis[i] <= 6)[0])
        for i in range(self.CA_cands.shape[0]):
            self.neighbors0to7.append(np.where(self.cand_self_dis[i] <= 7)[0])
        for i in range(self.CA_cands.shape[0]):
            self.neighbors2to7.append(np.where((self.cand_self_dis[i] <= 7) * (self.cand_self_dis[i] >= 2))[0])

        self.neigh_mat = np.zeros_like(self.cand_self_dis)
        for cand in range(self.CA_cands.shape[0]):
            for neigh in self.neighbors2to6[cand]:
                BB_dens = 0
                dis = max(0,abs(self.cand_self_dis[cand,neigh] - 3.8)-0.5)
                dis_score = max(0, 1 - dis / 2)
                for j in range(1,5):
                    coord=np.round(j/5*self.CA_cands[neigh]+(5-j)/5*self.CA_cands[cand]).astype(int)
                    BB_dens+=NNPred.BBProb[coord[0],coord[1],coord[2]]
                self.neigh_mat[cand,neigh]=(dis_score+BB_dens/4)/2
        
        self.best_neigh=[]
        for cand in range(self.CA_cands.shape[0]):
            neigh_list=[]
            second,first=self.neigh_mat[cand].argsort()[-2:]
            if self.neigh_mat[cand,first]!=0:
                neigh_list.append(first)
            if self.neigh_mat[cand,second]!=0:
                neigh_list.append(second)
            self.best_neigh.append(neigh_list)


    def checkSeq(self):
        print(self.dynamic_config.fasta_path)
        if os.path.exists(self.dynamic_config.fasta_path):
            fasta_lines=open(self.dynamic_config.fasta_path).readlines()
        else:
            return 'fasta not found!'
        un_exist_list=[]
        if self.dynamic_config.protocol=='temp_flex':
            print('You are using EModelX(+AF).')
            print('We will replace your fasta sequence as AlphaFold sequence (or the most similar one in AFDB).')
        
        input_list=[]
        fasta_set=set()
        seq=''
        for line_n, line in enumerate(fasta_lines):
            line=line.strip()
            if line.startswith('>'):
                head = line
                split_fasta=line[1:].split('|')[0]
                fasta_name=split_fasta
                n=0
                while fasta_name in fasta_set:
                    n+=1
                    fasta_name = f'{split_fasta}_{n}'
                fasta_set.add(fasta_name)
                seq=''
            else:
                seq=seq+line
            
            if line_n >=len(fasta_lines)-1 or fasta_lines[line_n+1].startswith('>'):
                if len(line)<10:
                    continue
                for i,c in enumerate(seq):
                    if c not in utils.AA_abb and c not in ['A','U','T','G','C']:
                        seq=seq[:i]+'A'+seq[i+1:]
                        print(f'warning!: {c} in protein seuqence would be treat as ALA')
                if ('U' in seq) or set(seq).issubset(set(['A','U','T','G','C'])):
                    continue
                seq_obj=Sequence(fasta_name, seq)
                try:
                    chain_strs = head.split('|')[1].split(',')
                except:
                    chain_strs=[random.choice(utils.chainID_list)]
                    print(f'WARNING!!! Parse chain number error!!! Chain number will be set as 1 for fasta_name \"{fasta_name}\"')
                    print(f'WARNING!!! Parse chain number error!!! Assigning chain id {chain_strs[0]} for fasta_name \"{fasta_name}\"')
                input_list.append((seq_obj,chain_strs))
            
        if input_list:
            print('AFDB searching...')
            print('The process typically requires between 5 to 10 minutes to complete when no same search hits before.')
            print('Waiting over 1000s would be recognized as search failed')
            pool = Pool(min(len(input_list),self.dynamic_config.mul_proc_num))
            results = pool.map(get_seq, input_list)

            for res in results:
                seq_obj,chain_strs,un_exist=res
                fasta_name=seq_obj.fasta_name
                if un_exist:
                    un_exist_list.append(fasta_name)

                for chain_str in chain_strs:
                    try:
                        chain_id = chain_str.split(' ')[-1].split(']')[0]
                    except:
                        chain_id='A'
                        print(f'WARNING!!! Parse chain ID error!!! Chain ID will be set as A for {chain_str}')
                    if fasta_name not in self.fastas:
                        self.fasta_list.append(fasta_name)
                        self.fastas[fasta_name] = seq_obj
                    new_chain_id=chain_id
                    if new_chain_id not in utils.chainID_list:
                        new_chain_id = random.choice(utils.chainID_list)
                    chain_n=0
                    while new_chain_id in self.fastas[fasta_name].chain_dict and chain_n<100:
                        chain_n+=1
                        new_chain_id = random.choice(utils.chainID_list)
                    if chain_id!=new_chain_id:
                        print(f'Wrong chain id! use random chain id {new_chain_id}!')
                    chain_id=new_chain_id
                    self.fastas[fasta_name].chain_dict[chain_id]=Chain(chain_id, seq_obj.sequence)
                    self.chain_id_list.append(chain_id)
                    self.max_seq_len=max(self.max_seq_len,len(seq_obj.sequence))
                    self.ResNum +=len(seq_obj.sequence)


        if len(self.fastas)==0:
            return 'Error in parse fasta, terminated!'
        if un_exist_list:
            return 'Templates not found for {}, Check your directory of templates!'.format(un_exist_list)
        
        print('Check your fasta inputs, the fasta parser can only handle standard fasta inputs and only protein sequences would be parsed:')
        for i, fasta_name in enumerate(self.fastas):
            print(f'Your sequence {i+1} name: {fasta_name}',end=', ')
            print(f'Chains that belong to this sequence:')
            for chain_id in self.fastas[fasta_name].chain_dict:
                print(chain_id,end=' ')
            print()
            print('Your sequence: ')
            print(self.fastas[fasta_name].sequence)
        return 'success'


    def prepareSeq4Align(self):
        self.seq_cand_AA_mat = np.zeros([len(self.fastas), self.max_seq_len, self.CA_cands.shape[0]]).astype(float)
        for i, fasta_name in enumerate(self.fastas):
            for j, AA in enumerate(self.fastas[fasta_name].sequence):
                for k, coord in enumerate(self.CA_cands):
                    if AA in utils.AA_abb:
                        self.seq_cand_AA_mat[i,j,k] = self.CA_cands_AAProb[utils.AA_abb[AA],k]
        
        


    def seqStructureAlign(self):
        self.n_hop_mat = self.getNHopMat()

        connect_len=5
        self.seq_cand_AA_mat_copy=self.seq_cand_AA_mat.copy()
        self.quasiSeqAlign(connect_len=connect_len)
        if not self.alignedFrags:
            return False

        connect_len=9
        self.seq_cand_AA_mat_copy[np.where(self.cand_match_result > 0)] = 1
        self.quasiSeqAlign(connect_len=connect_len)

        if not self.alignedFrags:
            return False
        return True


    def getNHopMat(self):
        n_hop_mat = np.zeros([self.dynamic_config.n_hop,self.cand_self_dis.shape[0],self.cand_self_dis.shape[0]])
        global globalParam
        globalParam.update(self.seq_cand_AA_mat, self.neigh_mat,neighbors2to6=self.best_neigh)
        pool = Pool(self.dynamic_config.mul_proc_num)
        async_results = []
        for cand in range(self.CA_cands.shape[0]):
            async_results.append(pool.apply_async(pathWalking, args=(cand,self.dynamic_config.n_hop)))
        pool.close()
        pool.join()

        for cand, async_result in enumerate(async_results):
            results=async_result.get()
            for n, res in enumerate(results):
                n_hop_mat[n,cand]=res
        for n in range(n_hop_mat.shape[0]):
            for cand in range(n_hop_mat.shape[1]):
                this_sum = np.sum(n_hop_mat[n, cand])
                if this_sum != 0:
                    n_hop_mat[n, cand] /= this_sum
        return n_hop_mat


    def quasiSeqAlign(self,connect_len):
        self.seq_align_score=self.seq_cand_AA_mat_copy.copy()
        for i in range(self.dynamic_config.n_hop):
            self.seq_align_score+=np.pad(self.seq_cand_AA_mat_copy[:,:-i-1,:],[(0,0),(i+1,0),(0,0)],constant_values=(0,0))@self.n_hop_mat[i].T+np.pad(self.seq_cand_AA_mat_copy[:,i+1:,:],[(0,0),(0,i+1),(0,0)],constant_values=(0,0))@self.n_hop_mat[i].T

        self.seq_cand_AA_mat_copy=self.seq_cand_AA_mat.copy()
        sort_ix = (-self.seq_align_score.max(axis=0).max(axis=0)).argsort()
        self.alignedFrags=[[] for _ in range(len(self.fastas))]
        self.cand_match_result=np.zeros_like(self.seq_cand_AA_mat_copy)
        used_cands=set()
        for cand_ix in sort_ix:
            if cand_ix in used_cands:
                continue
            fasta_ix,seq_ix = np.unravel_index(self.seq_align_score[:,:,cand_ix].argmax(),self.seq_align_score.shape[:2])
            if self.seq_align_score[fasta_ix,seq_ix,cand_ix]<=self.dynamic_config.score_thrh:
                continue
            fragment=self.findAlignedFrag(fasta_ix,seq_ix,cand_ix)
            if len(fragment[0])>=connect_len and np.mean(fragment[2]) > self.dynamic_config.score_thrh/2:
                self.alignedFrags[fasta_ix].append(fragment)
                for i, cand in enumerate(fragment[0]):
                    used_cands.add(cand)
                    self.cand_match_result[fasta_ix,fragment[1][i],cand] = fragment[2][i]
                    self.seq_align_score[:,:,cand]=0
                    self.seq_cand_AA_mat_copy[:,:,cand]=0
                    if np.sum(self.cand_match_result[fasta_ix,fragment[1][i]]>0) >= len(self.fastas[self.fasta_list[fasta_ix]].chain_dict):#remove it could be better but may cause bugs
                        self.seq_align_score[fasta_ix,fragment[1][i],:]=0
                        self.seq_cand_AA_mat_copy[fasta_ix,fragment[1][i],:] = 0
    

    def findAlignedFrag(self,fasta_ix,seq_ix,cand_ix):
        traces=[[cand_ix]]
        seqs = [[seq_ix]]
        scores=[[self.seq_align_score[fasta_ix,seq_ix,cand_ix]]]
        left_seq = seq_ix
        right_seq=seq_ix
        left_val=left_seq>0
        right_val = right_seq<len(self.fastas[self.fasta_list[fasta_ix]].sequence)-1

        max_scores = self.seq_align_score.max(axis=1)
        while left_val or right_val:
            if left_val:
                left_seq=left_seq-1
                left_val=left_seq>0
                tmp_traces=[]
                tmp_seqs=[]
                tmp_scores=[]
                for i,trace in enumerate(traces):
                    for neigh in self.neighbors0to7[trace[0]]:
                        if self.seq_align_score[fasta_ix,left_seq,neigh] == max_scores[fasta_ix,neigh]>self.dynamic_config.score_thrh:
                            tmp_traces.append([neigh]+trace)
                            tmp_seqs.append([left_seq]+seqs[i])
                            tmp_scores.append([self.neigh_mat[neigh,trace[0]]*self.seq_align_score[fasta_ix,left_seq,neigh]]+scores[i])
                if not tmp_traces:
                    left_val=False
                    left_seq += 1
                elif len(tmp_traces)>1:
                    max_ix=None
                    max_score=0
                    for i,trace in enumerate(tmp_traces):
                        score_sum = np.sum(tmp_scores[i])
                        if np.sum(tmp_scores[i])>max_score:
                            max_score=np.sum(score_sum)
                            max_ix =i
                    traces=[tmp_traces[max_ix]]
                    seqs=[tmp_seqs[max_ix]]
                    scores=[tmp_scores[max_ix]]
                

                else:
                    traces=tmp_traces
                    seqs=tmp_seqs
                    scores=tmp_scores

            if right_val:
                right_seq=right_seq+1
                right_val = right_seq<len(self.fastas[self.fasta_list[fasta_ix]].sequence)-1
                tmp_traces=[]
                tmp_seqs=[]
                tmp_scores=[]
                for i,trace in enumerate(traces):
                    for neigh in self.neighbors0to7[trace[-1]]:
                        if self.seq_align_score[fasta_ix,right_seq,neigh]== max_scores[fasta_ix,neigh]>self.dynamic_config.score_thrh:
                            tmp_traces.append(trace+[neigh])
                            tmp_seqs.append(seqs[i]+[right_seq])
                            tmp_scores.append(scores[i]+[self.neigh_mat[trace[-1],neigh]*self.seq_align_score[fasta_ix,right_seq,neigh]])
                if not tmp_traces:
                    right_val=False
                    right_seq -= 1
                elif len(tmp_traces)>1:
                    max_ix=None
                    max_score=0
                    for i,trace in enumerate(tmp_traces):
                        score_sum = np.sum(tmp_scores[i])
                        if np.sum(tmp_scores[i])>max_score:
                            max_score=np.sum(score_sum)
                            max_ix =i
                    traces=[tmp_traces[max_ix]]
                    seqs=[tmp_seqs[max_ix]]
                    scores=[tmp_scores[max_ix]]
                else:
                    traces=tmp_traces
                    seqs=tmp_seqs
                    scores=tmp_scores

        max_ix=None
        max_score=0
        for i,trace in enumerate(traces):
            score_sum = np.sum(scores[i])
            if np.sum(scores[i])>max_score:
                max_score=np.sum(score_sum)
                max_ix =i
        if max_ix is not None:
            return [traces[max_ix], seqs[max_ix], scores[max_ix]]
        else:
            return [[], [], []]


    def seqRegisterScoring(self):
        self.seqStructScoring()
        result_list=[]
        for fasta_ix in range(len(self.fastas)):
            result=[]
            chain_num=len(self.fastas[self.fasta_list[fasta_ix]].chain_dict)
            for i in range(chain_num):
                chain_result=[0 for _ in range(len(self.fastas[self.fasta_list[fasta_ix]].sequence))]
                result.append(np.array(chain_result))
            result_list.append(result)


        seq_struct_align_score_copy=self.seq_struct_align_score.copy()
        self.registerScores=[]
        for fasta_ix,fasta_name in enumerate(self.fastas):
            
            this_fasta=self.fastas[fasta_name]
            seq_len=len(this_fasta.sequence)
            chain_num=len(this_fasta.chain_dict)

            global globalParam
            globalParam.update(seq_struct_align_score_copy, self.neigh_mat,fastas=self.fastas,CA_cands=self.CA_cands,neighbors2to6=self.neighbors2to6,CAProb=self.CAProb)

            pool = Pool(self.dynamic_config.mul_proc_num)
            async_results = []
            for seq_ix in range(self.dynamic_config.struct_len//2+1,seq_len-self.dynamic_config.struct_len//2-1):
                async_results.append(pool.apply_async(registerScoring, args=(fasta_ix,fasta_name,seq_ix,self.dynamic_config.struct_len//2+1)))
            pool.close()
            pool.join()

            max_score=0
            for async_result in async_results:
                result=async_result.get()
                if len(result)>=chain_num and result[chain_num-1][0] > max_score:
                    max_score=result[chain_num-1][0]
            self.registerScores.append(max_score)
    

    def seqStructScoring(self):
        self.n_hop_mat = self.getNHopMat()

        self.local_traces=[]

        for cand in range(self.CA_cands.shape[0]):
            trace_dict={}
            traces=[[cand]]
            scores=[0]
            for i in range(self.dynamic_config.struct_len-1):
                tmp_traces=[]
                tmp_scores=[]
                for j,trace in enumerate(traces):
                    for nei in set(self.best_neigh[trace[-1]]) - set(trace):
                        tmp_traces.append(trace+[nei])
                        tmp_scores.append(scores[j]+self.neigh_mat[trace[-1],nei])
                traces= tmp_traces
                scores=tmp_scores
            for j,trace in enumerate(traces):
                if trace[-1] not in trace_dict or scores[j]> trace_dict[trace[-1]][1]:
                    if scores[j]/(self.dynamic_config.struct_len-1)>0.7:
                        trace_dict[trace[-1]]=[trace,scores[j]]
            for key in trace_dict:
                self.local_traces.append(trace_dict[key][0])
        print('traces: ',len(self.local_traces))
        assert(self.local_traces)
        self.struct_match = np.zeros_like(self.seq_cand_AA_mat)
        global globalParam
        globalParam.update(self.seq_cand_AA_mat,self.neigh_mat,fastas=self.fastas,CA_cands=self.CA_cands,trace_list=self.local_traces)


        for fasta_ix, fasta_name in enumerate(self.fastas):#并行
            chain_num=len(self.fastas[fasta_name].chain_dict)
            async_results=[]
            pool = Pool(self.dynamic_config.mul_proc_num)
            for start_j in range(len(self.fastas[fasta_name].sequence)-self.dynamic_config.struct_len+1):
                seq= range(start_j,start_j+self.dynamic_config.struct_len)
                async_results.append(pool.apply_async(localSeqStructAlign, args=(fasta_ix,fasta_name, seq)))
            pool.close()
            pool.join()
            align_results = []
            for async_result in async_results:
                align_results.append(async_result.get())

            for start_j, align_result in enumerate(align_results):
                seq= range(start_j,start_j+self.dynamic_config.struct_len)
                for t,score_list in enumerate(align_result):
                    AA_score,nei_score,af2_rmsd=score_list
                    score = AA_score+nei_score - min(1,max(0,af2_rmsd-1))**2
                    for i, s in enumerate(seq):
                        self.struct_match[fasta_ix,s,self.local_traces[t][i]]=max(self.struct_match[fasta_ix,s,self.local_traces[t][i]],score)

        self.struct_match[self.struct_match<0.1]=0.1

        self.struct_match_copy=self.struct_match.copy()
        self.seq_struct_align_score=self.struct_match_copy.copy()
        for i in range(self.dynamic_config.n_hop):
            self.seq_struct_align_score+=np.pad(self.struct_match_copy[:,:-i-1,:],[(0,0),(i+1,0),(0,0)],constant_values=(0,0))@self.n_hop_mat[i].T+np.pad(self.struct_match_copy[:,i+1:,:],[(0,0),(0,i+1),(0,0)],constant_values=(0,0))@self.n_hop_mat[i].T


    def registerExpand(self, chains, fasta_ix):
        fasta_name=self.fasta_list[fasta_ix]
        this_fasta=self.fastas[fasta_name]
        seq_len=len(this_fasta.sequence)
        chain_num=len(this_fasta.chain_dict)
        score4sort=[]
        for chain in chains:
            score4sort.append(chain[0])
        score_sort_ix =np.argsort(score4sort)[::-1]
        results=[]
        for j in score_sort_ix:
            score,this_trace,seq,_ = chains[j]
            this_seq = list(seq)

            left_seq = this_seq[0]
            right_seq = this_seq[-1]
            left_val = left_seq>0
            right_val = right_seq < seq_len-1
            while left_val or right_val:
                if left_val:
                    check_len = min(len(this_trace),20)
                    this_coords=self.CA_cands[this_trace[:check_len]]
                    this_AF2_coords=this_fasta.AF2_struct[left_seq:left_seq+check_len]
                    rmsd,R,T,_ = superpose3d.Superpose3D(this_coords,this_AF2_coords)
                    old_rmsd=rmsd[0]
                    trans_AF2 = np.dot(this_fasta.AF2_struct, R.T)+T
                    dis_AF2_trace = np.sqrt(np.sum((self.CA_cands-trans_AF2[left_seq-1])**2, axis=1))
                    if old_rmsd < 5 and dis_AF2_trace.min()<3:
                        left_seq-=1
                        this_trace=[dis_AF2_trace.argmin()]+this_trace
                        left_val = left_seq>0
                    else:
                        left_val = False

                if right_val:
                    check_len = min(len(this_trace),20)
                    this_coords=self.CA_cands[this_trace[-check_len:]]
                    this_AF2_coords=this_fasta.AF2_struct[right_seq-check_len+1:right_seq+1]
                    rmsd,R,T,_ = superpose3d.Superpose3D(this_coords,this_AF2_coords)
                    old_rmsd=rmsd[0]
                    trans_AF2 = np.dot(this_fasta.AF2_struct, R.T)+T
                    dis_AF2_trace = np.sqrt(np.sum((self.CA_cands-trans_AF2[right_seq+1])**2, axis=1))
                    if old_rmsd < 5 and dis_AF2_trace.min()<3:
                        right_seq+=1
                        this_trace=this_trace+[dis_AF2_trace.argmin()]
                        right_val = right_seq < seq_len-1
                    else:
                        right_val = False

            this_seq= [_ for _ in range(left_seq,right_seq+1)]
            rmsd,R,T,_ = superpose3d.Superpose3D(self.CA_cands[this_trace],this_fasta.AF2_struct[this_seq])
            trans_AF2=np.dot(this_fasta.AF2_struct, R.T)+T
            trans_AF2 = np.round(trans_AF2).astype(int)
            trans_AF2 = trans_AF2[np.where(np.sum(trans_AF2>=0,axis=1)==3)]
            trans_AF2 = trans_AF2[np.where(np.sum(trans_AF2 < self.CAProb.shape, axis=1) == 3)]
            CA_prob_sum = np.sum(self.CAProb[trans_AF2[:, 0], trans_AF2[:, 1], trans_AF2[:, 2]])
            results.append([this_seq, this_trace,CA_prob_sum])
        return results
 

    def fillGap(self,fasta_ix,start_end):
        fasta_name=self.fasta_list[fasta_ix]
        this_fasta = self.fastas[fasta_name]
        seq_len = len(this_fasta.sequence)
        chain_list=list(this_fasta.chain_dict.keys())
        this_chain_id=chain_list[start_end[0]]
        left_pos = start_end[2]
        right_pos = start_end[3]
        final_seq=range(start_end[2],start_end[3]+1)
        left_val=True
        right_val=True
        dir = 1
        if left_pos==-1 and right_pos==seq_len:
            return
        elif left_pos==-1:
            left_traces = []
            right_traces = [[this_fasta.chain_dict[this_chain_id].result[right_pos]]]
            left_infos = []
            right_infos=[[[],[],0]]
            left_val=False
            left_seq = []
            right_seq=  [right_pos]
            dir = -1
        elif right_pos==seq_len:
            left_traces = [[this_fasta.chain_dict[this_chain_id].result[left_pos]]]
            right_traces = []
            left_infos = [[[],[],0]]
            right_infos=[]
            left_seq = [left_pos]
            right_seq=  []
            right_val=False
        else:
            left_traces = [[this_fasta.chain_dict[this_chain_id].result[left_pos]]]
            right_traces = [[this_fasta.chain_dict[this_chain_id].result[right_pos]]]
            left_infos = [[[],[],0]]
            right_infos=[[[],[],0]]
            left_seq = [left_pos]
            right_seq=  [right_pos]
        
        while (left_val or right_val) and left_pos!=right_pos and left_pos<len(this_fasta.sequence)-1 and right_pos>0:
            if dir == 1:
                this_traces = left_traces
                this_infos= left_infos
                left_pos+=dir
                end=-1
                this_seq = left_seq+[left_pos]
                this_pos = left_pos
            else:
                this_traces = right_traces
                this_infos= right_infos
                right_pos+=dir
                end=0
                this_seq = [right_pos]+right_seq
                this_pos = right_pos

            matched_chain=[[],[]]
            if self.dynamic_config.protocol=='temp_flex':
                matched_chain=[range(len(this_seq)),this_fasta.AF2_struct[this_seq]]
            else:
                max_len=5
                for chain_id in this_fasta.chain_dict:
                    matched_pos=[]
                    matched_coords=[]
                    for p,pos in enumerate(this_seq):
                        if this_fasta.chain_dict[chain_id].result[pos]!= -1:
                            matched_pos.append(p)
                            cand= this_fasta.chain_dict[chain_id].result[pos]
                            matched_coords.append(self.CA_cands[cand])
                    if len(matched_pos)>max_len:
                        matched_chain=[matched_pos,matched_coords]
                        max_len=len(matched_pos)

            tmp_traces=[]
            tmp_infos=[]
            tmp_scores=[]
            for ix, trace in enumerate(this_traces):
                if len(trace)-len(set(trace))>max(5,len(trace)//10):
                    continue
                this_info = this_infos[ix]
                cand = trace[-1] if dir == 1 else trace[0]
                nei_list=list(set(self.neighbors2to6[cand])-self.used_cands-set(trace))
                for neigh in nei_list:
                    new_trace = trace+[neigh] if dir == 1 else [neigh]+trace
                    
                    cand_score=this_info[0] + [this_fasta.chain_cand_mat[start_end[0], this_pos, neigh]]
                    neigh_score=this_info[1]+[self.neigh_mat[cand,neigh]]
                    sym_score = this_info[2]
                    if len(this_seq) >3 and len(this_seq)-1 in matched_chain[0]:
                        this_coords=[]
                        for p in matched_chain[0]:
                            this_coords.append(self.CA_cands[new_trace[p]])
                        sym_score = max(0,superpose3d.Superpose3D(this_coords,matched_chain[1])[0][0]-1)/2
                    
                    score = np.mean(np.array(cand_score)+np.array(neigh_score))- sym_score
                    tmp_traces.append(new_trace)
                    tmp_infos.append([cand_score, neigh_score, sym_score])
                    tmp_scores.append(score)

                    
            if not tmp_traces:
                if dir ==1:
                    left_val=False
                    dir*=-1
                    continue
                else:
                    right_val=False
                    dir*=-1
                    continue
            
            elif len(tmp_traces)>1000 or right_pos-left_pos<=2:
                this_traces=[]
                this_infos=[]
                last_dict={}
                max_score=-np.inf
                max_last=None
                for ix, trace in enumerate(tmp_traces):
                    if trace[end] not in last_dict or tmp_scores[ix] > last_dict[trace[end]][1]:
                        last_dict[trace[end]] = [trace, tmp_scores[ix],tmp_infos[ix]]
                        if tmp_scores[ix] > max_score:
                            max_score=tmp_scores[ix]
                            max_last=trace[end]

                for last in last_dict:
                    if self.cand_self_dis[last,max_last]<20:
                        this_traces.append(last_dict[last][0])
                        this_infos.append(last_dict[last][2])

                if dir == 1:
                    left_seq = left_seq+[left_pos]
                else:
                    right_seq=[right_pos]+right_seq
            else:
                if dir == 1:
                    left_seq = left_seq+[left_pos]
                else:
                    right_seq=[right_pos]+right_seq
                this_traces = tmp_traces
                this_infos = tmp_infos

            if dir ==1:
                left_traces= this_traces
                left_infos = this_infos
                
            else:
                right_traces= this_traces
                right_infos=this_infos

            if left_val and right_val:
                dir*=-1

        max_trace=None
        max_score=-np.inf
        
        if left_traces and right_traces and len(left_traces[0])+len(right_traces[0])-1==len(final_seq):
            
            for il, left_trace in enumerate(left_traces):
                for ir, right_trace in enumerate(right_traces):
                    if left_trace[-1]==right_trace[0]:
                        left_score=np.mean(np.array(left_infos[il][0])+np.array(left_infos[il][1]))- left_infos[il][2]
                        right_score=np.mean(np.array(right_infos[ir][0])+np.array(right_infos[ir][1]))- right_infos[ir][2]
                        if left_score+right_score > max_score:
                            max_trace = left_trace+right_trace[1:]
                            max_score = left_score+right_score

            if max_trace!= None:
                used_cands=set()
                for p in range(len(final_seq)//2+1):
                    left_pos=list(final_seq)[p]
                    right_pos=list(final_seq)[-p-1]
                    if max_trace[p] not in used_cands:
                        used_cands.add(max_trace[p])
                        this_fasta.chain_dict[this_chain_id].result[left_pos]=max_trace[p]
                    if max_trace[-p-1] not in used_cands:
                        used_cands.add(max_trace[-p-1])
                        this_fasta.chain_dict[this_chain_id].result[right_pos]=max_trace[-p-1]
        if max_trace is None:
            
            max_left_trace=None
            max_left_score=-np.inf
            for il, left_trace in enumerate(left_traces):
                left_score = np.mean(np.array(left_infos[il][0])+np.array(left_infos[il][1]))- left_infos[il][2]
                if left_score > max_left_score:
                    max_left_trace = left_trace
                    max_left_score = left_score
                
            max_right_trace=None
            max_right_score=-np.inf
            for ir, right_trace in enumerate(right_traces):
                right_score = np.mean(np.array(right_infos[ir][0])+np.array(right_infos[ir][1]))- right_infos[ir][2]
                if right_score > max_right_score:
                    max_right_trace = right_trace
                    max_right_score = right_score

            gap = 0
            if max_left_trace is not None and max_right_trace is not None:
                gap = max(0,(self.cand_self_dis[max_left_trace[-1],max_right_trace[0]] - 3 * (right_pos-left_pos))) // 6
            # print(gap)
            if max_left_trace is not None:
                # print(self.GT_cand_dis[fasta_ix,:,left_seq,max_left_trace].min(axis=1))
                for p in range(len(left_seq)-int(gap)):
                    left_pos=list(left_seq)[p]
                    this_fasta.chain_dict[this_chain_id].result[left_pos]=max_left_trace[p]
                
            if max_right_trace is not None:
                # print(self.GT_cand_dis[fasta_ix,:,right_seq,max_right_trace].min(axis=1))
                for p in range(int(gap),len(right_seq)):
                    right_pos=list(right_seq)[p]
                    this_fasta.chain_dict[this_chain_id].result[right_pos]=max_right_trace[p]
  