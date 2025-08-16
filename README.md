# E3-CryoFold: End-to-end Prediction For Cryo-EM Structure Determination(updated)

E3-CryoFold is a deep learning framework for automating the determination of three-dimensional atomic structures from high-resolution cryo-electron microscopy (Cryo-EM) density maps. It addresses the limitations of existing AI-based methods by providing an end-to-end solution that integrates training and inference into a single streamlined pipeline. E3-CryoFold combines 3D and sequence Transformers for feature extraction and employs an equivariant graph neural network to build accurate atomic structures from density maps.

<p align="center" width="100%">
  <img src='https://github.com/user-attachments/assets/accbe5f4-a2de-46f7-8255-8c36106770a5' width="100%">
</p>

## Table of Contents
- [Background](#background)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Command-Line Arguments](#command-line-arguments)
  - [Running the Example](#running-the-example)
  - [Using Custom Data](#using-custom-data)
- [Tutorial](#tutorial)
- [References](#references)
- [Contact](#contact)
- [License](#license)

## Background

Cryo-electron microscopy (Cryo-EM) has revolutionized structural biology by enabling the visualization of complex biological molecules at near-atomic resolution. The technique generates **high-resolution density maps** that offer insights into the molecular structures of proteins, viruses, and other biomolecular assemblies. However, **interpreting these density maps to derive accurate atomic models** remains a challenging and labor-intensive task, often requiring expert knowledge and manual interventions.

Existing AI-based methods for automating Cryo-EM structure determination face several limitations:
1. **Multi-stage processing**: Current approaches often involve separate stages for feature extraction, sequence alignment, and structure prediction, leading to inefficiencies and discontinuities.
2. **Alignment bias**: Techniques such as **Hidden Markov Models (HMMs)** or **Traveling Salesman Problem (TSP) solvers** introduce bias when aligning predicted atomic coordinates with the protein sequence.
3. **Poor generalization**: Due to the limited size of available datasets, many methods struggle to generalize well to complex or previously unseen test cases.

E3-CryoFold addresses these challenges by providing a **fully integrated, end-to-end solution** that performs **one-shot inference** with minimal manual intervention, enabling faster and more accurate structure determination.

## Updated Version
To solve the issues reported by users, we updated E3-CryoFold from following aspects:
- **Enhanced Spatial Constraints**: To further improve the generalizability and stability of E3-CryoFold, instead of using the resized density map as spatial counterpart, we introduce to use the Cα atoms predicted from the density maps as spatial features. This approach constrains the generated structures to better fit to the density maps, which has been discussed in the future work section of our paper.
- **Extended Sequence Modeling**: To reduce computational costs, we integrated spatial-sequential modeling into the SE(3)-GNN framework. This enhancement allows E3-CryoFold to generate longer chains in a one-shot manner.
- **Support for Multiple Configurations**: The updated E3-CryoFold supports a wider range of settings for diverse scenarios, including `sequence-free`, `sequence-free de novo`, `pre-alignment`, and `de novo` settings.

## Installation

To get started with E3-CryoFold, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/A4Bio/E3-CryoFold.git
   cd E3-CryoFold
   ```

2. **Create and activate the conda environment**:

  ```bash
  conda create -n e3cryofold python=3.9
  conda activate e3cryofold
  bash install.sh
  ```

  ***Note: If you encounter issues with the installation of `torch-scatter`/`torch-geometric`/`torch-cluster`, you can install these softwares using the following command:***

   ```bash
  pip install torch-scatter/torch-geometric/torch-cluster -f https://data.pyg.org/whl/torch-(your-pytorch-version)+cu(your_CUDA_version).html
   ```

  For example, if your pytorch version is 2.1.1 and the CUDA version is 11.8. The installation commend is:
   ```bash
    pip install torch-scatter/torch-geometric/torch-cluster -f https://data.pyg.org/whl/torch-2.1.1+cu118.html
   ```

  The detailed information can refer to [https://github.com/rusty1s/pytorch_scatter?tab=readme-ov-file](https://github.com/rusty1s/pytorch_scatter?tab=readme-ov-file)

3. **Download the Pretrained Model**:

    We provide a pretrained model for E3-CryoFold. [Download it here](https://github.com/A4Bio/E3-CryoFold/releases/download/checkpoint/models.zip) and place it in the models directory.

4. **Download the Experimental dataset**:
   The training set can be downloaded in https://doi.org/10.7910/DVN/FCDG0W, and the standard test dataset can be downloaded in https://doi.org/10.7910/DVN/2GSSC9.


## Quick Start

To quickly try out E3-CryoFold using an example dataset, run the following command:

```
bash run_example.sh
```

This script runs the `inference.py` script with sample data provided in the `data/inputs` folder.


## Usage

### Command-line Arguments

The `inference.py` script supports several command-line arguments:

| Argument                 | Description                                             | Default                             |
|--------------------------|---------------------------------------------------------|-------------------------------------|
| `--map_path`             | Path to the input density map directory (required).     | None                                |
| `--fasta_path`           | Path to the fasta file (optional, if you set protocol is seq_free).           | None                       |
| `--pretrained`           | Path to the pretrained model checkpoint.                | `models/model_weight.pth`                       |
| `--stru_pretrained`      | Path to the structure pretrained model checkpoint.                | `./models/structure_model.pth`       |
| `--save_dir`             | Directory to save the output PDB file.                  | `./data/outputs/`             |
| `--save_name`            | Name to save the output PDB file.                       | `example`.                 |
| `--protocol`             | Setting to generate the atomic structure.               | `pre_align`.               |
| `--device`               | Device to run the model on (`cpu` or `cuda`).           | `cuda`                     |
| `--spatial_condition`    | whether use spatial conditioner.                        | True                       |
| `--sequence_condition`   | whether use sequence conditioner.                       | False                      |
| `--t`                    | Reverse Stochastic Differential Equation(SDE) start time                                        | 0.1                   |


### Using Custom Data

To use E3-CryoFold with your own data, you need to provide a Cryo-EM density map and, optionally, a fasta file for evaluating the predicted structure. For example:

```bash
python inference.py --map_path /path/to/your/density_map --fasta_path /path/to/your/fasta --save_dir /path/to/save/results --save_name your_file_name
```

## Tutorial

### 1. Using pre-align setting:

To improve the constrain of the spatial feature for the generated structure, you can use the pre-align setting:

	$ python inference.py --map_path data/inputs/maps/emd_32336.map.gz --fasta_path data/inputs/fastas/7w72 --save_dir ./data/outputs/ --save_name 32336-7w72 --protocol pre_align --t 0.1


### 2. Using denovo setting:

To improve the constrain of the spatial feature for the generated structure, you can use the pre-align setting:

	$ python inference.py --map_path data/inputs/maps/emd_8623.map.gz --fasta_path data/inputs/fastas/5uz7 --save_dir ./data/outputs/ --save_name 8623-5uz7-denovo --protocol denovo --t 0.1

After inference, the output will be saved in the specified output directory:

```text 
E3-CryoFold
├── data
│   └── outputs
          ├── 8623-5uz7-denovo.pdb
          └── 8623-5uz7-denovo_all_atom_model.pdb
```
### 3. Enhancing the spatial constrain:

If you want to make the generated structure align more closely with the density map and more rational, consider increasing the argument `--t` and the SDE steps.

	$ python inference.py --map_path data/inputs/maps/emd_8623.map.gz --fasta_path data/inputs/fastas/5uz7 --save_dir ./data/outputs/ --save_name 8623-5uz7-denovo --protocol denovo --t 1.0

This argument increase the refinement steps for structures, but it may also introduce some structural bias.

### 4. Sequence-free modelling:
If you only have a density map without sequence, we also provide two settings to generate the sequence-free structures:

```bash
# pre-align without sequence
python inference.py --map_path data/inputs/maps/emd_8623.map.gz  --save_dir ./data/outputs/ --save_name 8623-5uz7-seqfree --protocol seq_free --t 0.1
```

```bash
# denovo without sequence
python inference.py --map_path data/inputs/maps/emd_8623.map.gz --save_dir ./data/outputs/ --save_name 8623-5uz7-seqfree-denovo --protocol seq_free_denovo --t 1.0
```



## Applications

#### Users can utilize E3-CryoFold for several applications:
### 1. Cryo-EM structure derermination
E3-CryoFold can generate structures with high accuracy. In some cases, the pre-align setting performs better, while in other scenarios, the de novo setting yields ideal and rational structures.

### 2. Cryo-EM structure refinement
E3-CryoFold incorporates the Stochastic Differential Equation (SDE) function with spatial and sequential information as constraints. This approach guides the structure to align more closely with the density map, resulting in more rational designs.

### 3. Low-resolution Cryo-EM structure design
The Cryo-EM density map with low resolution is hard to be solved by current AI method. Since E3-CryoFold fully leverage the sequential information and spatial constrain, E3-CryoFold can provide an effective way to design ideal atomic structure for templates or guidence. We recommend following commend for solving low-resolution map:

	$ python inference.py --map_path /path/to/your/density_map --fasta_path /path/to/your/fasta --save_dir /path/to/save/results --save_name your_file_name --protocol denovo --t 1.0 --spatial_conditon True --sequence_condition True



## References

```
Chen, S., Zhang, S., Fang, X. et al. Protein complex structure modeling by cross-modal alignment between cryo-EM maps and protein sequences. Nat Commun 15, 8808 (2024). https://doi.org/10.1038/s41467-024-53116-5

Ingraham, J.B., Baranov, M., Costello, Z. et al. Illuminating protein space with a programmable generative model. Nature 623, 1070–1078 (2023). https://doi.org/10.1038/s41586-023-06728-8
```

## Contact

Please submit any bug reports, feature requests, or general usage feedback as a github issue or discussion.

- Jue Wang (wangjue@westlake.edu.cn)
- Cheng Tan (tancheng@westlake.edu.cn)
- Zhangyang Gao (gaozhangyang@westlake.edu.cn)

## License

This project is licensed under the MIT License. See the [LICENSE file](https://github.com/A4Bio/E3-CryoFold/blob/main/LICENSE) for details.

