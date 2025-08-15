
# run pre-alignment inference
python inference.py --map_path data/inputs/maps/emd_32336.map.gz --fasta_path data/inputs/fastas/7w72 --save_dir ./data/outputs/ --save_name 32336-7w72 --protocol pre_align --t 0.1

# run denovo inference
python inference.py --map_path data/inputs/maps/emd_8623.map.gz --fasta_path data/inputs/fastas/5uz7 --save_dir ./data/outputs/ --save_name 8623-5uz7-denovo --protocol denovo --t 0.1