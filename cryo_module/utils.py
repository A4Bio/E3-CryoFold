import numpy as np
from skimage import transform
import numpy as np
import mrcfile
import os
import requests
import time
import random


chainID_list = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9']

AA_types = {"ALA":1,"CYS":2,"ASP":3,"GLU":4,"PHE":5,"GLY":6,"HIS":7,"ILE":8,"LYS":9,"LEU":10,"MET":11,"ASN":12,"PRO":13,"GLN":14,"ARG":15,"SER":16,"THR":17,"VAL":18,"TRP":19,"TYR":20}
AA_T = {AA_types[k]-1: k for k in AA_types}
AA_abb_T = {0:"A",1:"C",2:"D",3:"E",4:"F",5:"G",6:"H",7:"I",8:"K",9:"L",10:"M",11:"N",12:"P",13:"Q",14:"R",15:"S",16:"T",17:"V",18:"W",19:"Y"}
AA_abb = {AA_abb_T[k]:k for k in AA_abb_T}
abb2AA = {"A":"ALA","C":'CYS',"D":'ASP',"E":'GLU',"F":'PHE',"G":'GLY',"H":"HIS","I":"ILE","K":"LYS","L":"LEU","M":"MET","N":"ASN","P":"PRO","Q":"GLN","R":"ARG","S":"SER","T":"THR","V":"VAL","W":"TRP","Y":"TYR"}
AA2abb = {abb2AA[k]:k for k in abb2AA}
MMalign='~/projects/MMalign/MMalign' # your MMalign path
TMalign='~/projects/MMalign/TMalign' # your TMalign path

def get_info_from_csv(csv_data):
    emid = str(csv_data[0])
    date = str(csv_data[1])
    resol = float(csv_data[2])
    pdbid = str(csv_data[3])
    while len(emid) < 4:
        emid = '0' + emid
    return emid,date,resol,pdbid

def visualize(map,path,offset):
    with mrcfile.new(path,data=map.astype(np.float32), overwrite=True) as image_mrc:
        image_mrc.header.nzstart = offset[0]
        image_mrc.header.nystart = offset[1]
        image_mrc.header.nxstart = offset[2]
        image_mrc.header.maps = 1
        image_mrc.header.mapr = 2
        image_mrc.header.mapc = 3
    

def transpose(numpy_image, axis_order, offset):
    trans_offset = []
    trans_order = []
    for i in range(3):
        for j in range(len(axis_order)):
            if axis_order[j] == i:
                trans_offset.append(offset[j])
                trans_order.append(j)
    image = np.transpose(numpy_image, trans_order)

    return image, trans_offset


def reshape(numpy_image, offset, pixel_size):
    if pixel_size == [1,1,1]:
        return numpy_image, offset
    image = transform.rescale(numpy_image, pixel_size)
    for i in range(len(offset)):
        offset[i] *= pixel_size[i]
    return image, offset


def normalize(numpy_image,offset):
    np.nan_to_num(numpy_image)
    median = np.median(numpy_image)
    image = (numpy_image > median) * (numpy_image - median)
    
    vlid_coords = np.array(np.where(image>0))
    minX = np.min(vlid_coords[0])
    maxX = np.max(vlid_coords[0])
    minY = np.min(vlid_coords[1])
    maxY = np.max(vlid_coords[1])
    minZ = np.min(vlid_coords[2])
    maxZ = np.max(vlid_coords[2])
    image = image[minX:maxX+1,minY:maxY+1,minZ:maxZ+1]
    # print('origin shape',image.shape,'new shape', image.shape)
    minXYZ = [minX,minY,minZ]
    offset = [offset[0]+minXYZ[0], offset[1]+minXYZ[1], offset[2]+minXYZ[2]]
    
    p999 = np.percentile(image[np.where(image > 0)], 99.9)
    if p999 != 0:
        image = (image < p999) * image + (image >= p999) * p999
        image /= p999
        return image, offset
    else:
        print('normalization error!!!')
        return


def processEMData(EMmap,norm=True):
    em_data = np.array(EMmap.data)
    pixel_size = [float(EMmap.header.cella.x / EMmap.header.mx),float(EMmap.header.cella.y / EMmap.header.my),float(EMmap.header.cella.z / EMmap.header.mz)]
    axis_order = [int(EMmap.header.maps) - 1, int(EMmap.header.mapr) - 1,int(EMmap.header.mapc) - 1]
    offset = [float(EMmap.header.nzstart), float(EMmap.header.nystart),float(EMmap.header.nxstart)]
    # print(pixel_size,axis_order,offset,end='\t')
    em_data, offset = transpose(em_data, axis_order, offset)
    em_data, offset = reshape(em_data, offset, pixel_size)
    if norm:
        em_data, offset = normalize(em_data, offset)
    # offset=[offset[0]+float(EMmap.header.origin.x), offset[1]+float(EMmap.header.origin.y),offset[2]+float(EMmap.header.origin.z)]
    return em_data, offset


def calc_dis(coordList1,coordList2):
    y = [coordList2 for _ in coordList1]
    y = np.array(y)
    x = [coordList1 for _ in coordList2]
    x = np.array(x)
    x = x.transpose(1, 0, 2)
    a = np.linalg.norm(np.array(x) - np.array(y), axis=2)
    return a

def parseMMscore(gt_pdb,pred_pdb):
    lines = os.popen(f'{MMalign} \'{gt_pdb}\' \'{pred_pdb}\'').readlines()
    ResNum_pdb,ResNum_pred,Align_len,MM1,MM2,RMSD,SeqID=9999,9999,0,9999,9999,9999,9999
    try:
        for line in lines:
            if len(line)>len('Length of Structure_1') and line[:len('Length of Structure_1')]=='Length of Structure_1':
                ResNum_pdb=int(line.split(':')[1].split('residues')[0])
            elif len(line)>len('Length of Structure_2') and line[:len('Length of Structure_2')]=='Length of Structure_2':
                ResNum_pred=int(line.split(':')[1].split('residues')[0])
            elif len(line)>len('Aligned length=') and line[:len('Aligned length=')]=='Aligned length=':
                Align_len=float(line.split('Aligned length=')[1].split(',')[0])
                RMSD=float(line.split('RMSD=')[1].split(',')[0])
                SeqID=float(line.split('n_aligned=')[1].strip())
            elif line.find('normalized by length of Structure_1')!=-1:
                MM1=float(line.split('TM-score=')[1].split('(')[0])
            elif line.find('normalized by length of Structure_2')!=-1:
                MM2=float(line.split('TM-score=')[1].split('(')[0])
    except:
        pass
    return ResNum_pdb,ResNum_pred,Align_len,MM1,MM2,RMSD,SeqID

def parseChainCompscore(cmd):
    lines = os.popen(cmd).readlines()
    Close_RMSD, Close_N, Far_N, Close_Forward_N, Close_Reverse_N, Close_Mixed_N, Found, CA_Score, Seq_Match, Seq_Score, Mean_length, Fragments, Bad_Connections=-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
    try:
        s=lines[-1][14:].strip().split()
        Close_RMSD, Close_N, Far_N, Close_Forward_N, Close_Reverse_N, Close_Mixed_N, Found, CA_Score, Seq_Match, Seq_Score, Mean_length, Fragments, Bad_Connections=float(s[0]),int(s[1]),int(s[2]),int(s[3]),int(s[4]),int(s[5]),float(s[6]),float(s[7]),float(s[8]),float(s[9]),float(s[10]),int(s[11]),int(s[12])
    except:
        pass
    return Close_RMSD, Close_N, Far_N, Close_Forward_N, Close_Reverse_N, Close_Mixed_N, Found, CA_Score, Seq_Match, Seq_Score, Mean_length, Fragments, Bad_Connections

def parseTMscore(gt_pdb,pred_pdb):
    lines = os.popen(f'{TMalign} {gt_pdb} \'{pred_pdb}\'').readlines()
    ResNum_pdb,ResNum_pred,Align_len,MM1,MM2,RMSD,SeqID=9999,9999,0,9999,9999,9999,9999
    try:
        for line in lines:
            if len(line)>len('Length of Chain_1') and line[:len('Length of Chain_1')]=='Length of Chain_1':
                ResNum_pdb=int(line.split(':')[1].split('residues')[0])
            elif len(line)>len('Length of Chain_2') and line[:len('Length of Chain_2')]=='Length of Chain_2':
                ResNum_pred=int(line.split(':')[1].split('residues')[0])
            elif len(line)>len('Aligned length=') and line[:len('Aligned length=')]=='Aligned length=':
                Align_len=float(line.split('Aligned length=')[1].split(',')[0])
                RMSD=float(line.split('RMSD=')[1].split(',')[0])
                SeqID=float(line.split('n_aligned=')[1].strip())
            elif line.find('normalized by length of Chain_1')!=-1:
                MM1=float(line.split('TM-score=')[1].split('(')[0])
            elif line.find('normalized by length of Chain_2')!=-1:
                MM2=float(line.split('TM-score=')[1].split('(')[0])
    except:
        pass
    return ResNum_pdb,ResNum_pred,Align_len,MM1,MM2,RMSD,SeqID


def get_afdb_id_by_seq(seq,allow_seq_id=0.95):
    if len(seq)<10:
        return f'Error! Sequence len {len(seq)} small than 10!'
    jsonResponse={'message': 'Search in progress, please try after sometime!'}
    time_start=time.time()
    try_num=0
    error=False
    random_sleep=60
    
    while 'SeqId' not in jsonResponse and time.time()-time_start<1000:
        try:
            random_rows=int(round(random.random()*9))+1
            response = requests.get(f'https://alphafold.ebi.ac.uk/api/search?q={seq}&type=sequence&start=0&rows={random_rows}')
            response.raise_for_status()
            # access JSOn content
            jsonResponse = response.json()
            if 'docs' in jsonResponse:
                if 'entryId' in jsonResponse['docs'][0]:
                    if 'hsps'  in jsonResponse['docs'][0] and 'uniprotSequence' in jsonResponse['docs'][0]:
                        if 'identity' in jsonResponse['docs'][0]['hsps'] and 'subject' in jsonResponse['docs'][0]['hsps']:
                            seq_id=jsonResponse['docs'][0]['hsps']['identity']/len(seq)
                            af_id=jsonResponse['docs'][0]['entryId']
                            af_all_seq=jsonResponse['docs'][0]['uniprotSequence']
                            af_sub_seq=jsonResponse['docs'][0]['hsps']['subject'].replace('-','')
                            ind_start=af_all_seq.find(af_sub_seq)
                            ind_end=ind_start+len(af_sub_seq)
                            if ind_start==-1:
                                return 'Error! Can\'t find subject sequence in uniprotSequence!'

                            if seq_id>=allow_seq_id:
                                return {'AFDB entryId': af_id, 'Sequence Identity': seq_id,'AFDB subject sequence': af_sub_seq, 'AFDB subject sequence range': (ind_start,ind_end)}
                            else:
                                return f'Failed! AFID {af_id} with the max Seq Identity {seq_id} < {allow_seq_id}!'
                return f'Failed! Seq \n\t\"{seq}\"\n\tis not found in AFDB'
            
            if try_num>0:
                random_sleep=round(random.random()*90)+30
                print('Retry {:d} time'.format(try_num), flush=True)
            try_num+=1
            print('Please waiting for {:d}s....'.format(random_sleep), flush=True)
            time.sleep(random_sleep)
            print('In total {:d}s have past'.format(round(time.time()-time_start)), flush=True)
            error=False
        except Exception as ex:
            print('Error!: ',ex, flush=True)
            if error:
                return 'Failed! Continuous error'
            error=True
            try_num+=1
            print('Retry {:d} time'.format(try_num), flush=True)
            print(f'Continuous error would be recognized as search failed', flush=True)

            
    return 'Failed! Waiting for sequence search in AFDB over 1000s!'

def get_af_pdb_by_seq(seq,fasta_name,save_path,allow_seq_id):
    random_sleep=random.random()*5
    time.sleep(random_sleep)
    print(f'Your AFDB search is currently underway for {fasta_name}:\t\"{seq}\"', flush=True)
    
    res=get_afdb_id_by_seq(seq,allow_seq_id)
    random_sleep=random.random()*5
    time.sleep(random_sleep)
    print(res, flush=True)
    if not isinstance(res,dict) or 'AFDB entryId' not in res:
        print(f'{fasta_name} not found in AFDB!', flush=True)
        return '',None
    
    try:
        os.makedirs(os.path.dirname(save_path))
    except:
        pass
    af_id=res['AFDB entryId']
    af_sub_seq=res['AFDB subject sequence']
    af_sub_seq_range=res['AFDB subject sequence range']
    os.system(f'wget https://alphafold.ebi.ac.uk/files/{af_id}-model_v4.pdb -O {save_path}')
    if os.path.exists(save_path):
        return af_sub_seq, af_sub_seq_range
    else:
        print(f'{fasta_name} download from AFDB failed!', flush=True)
        return '',None