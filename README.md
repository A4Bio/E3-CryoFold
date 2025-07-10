### Note: We have recently received feedback regarding installation and modeling issues with our models. We are working on checking and resolving these issues; therefore, we temporarily removed the pretrained model. We apologize for the delay and any inconvenience caused by our code and pretrained model.

# E3-CryoFold: One-shot Prediction For Cryo-EM Structure Determination

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

## Features

- **🚀 End-to-End Training and Inference**: Simplifies the process by seamlessly integrating training and inference into a single, unified framework, eliminating the need for multi-stage processing.
- **⚡ Fast and Accurate**: Achieves a **400% improvement in TM-score** over Cryo2Struct while reducing inference time by a factor of **1,000**.

For more details on the performance and benchmarking, please refer to our paper.

## Installation

To get started with E3-CryoFold, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/A4Bio/E3-CryoFold.git
   cd E3-CryoFold
   ```

2. **Create and activate the conda environment**:

    ```bash
    conda env create -f environment.yml
    conda activate cryofold
    ```

3. **Download the Pretrained Model**:

    We provide a pretrained model for E3-CryoFold. [Download it here](https://github.com/A4Bio/E3-CryoFold/releases/download/checkpoint/checkpoint.pt) and place it in the pretrained_models directory （we ）.

4. **Download the Experimental dataset**:
   The training set can be downloaded in https://doi.org/10.7910/DVN/FCDG0W, and the standard test dataset can be downloaded in https://doi.org/10.7910/DVN/2GSSC9.


## Quick Start

To quickly try out E3-CryoFold using an example dataset, run the following command:

```
bash run_example.sh
```

This script runs the `inference.py` script with sample data provided in the `examples` folder. It uses a sample density map and a ground truth PDB file for evaluation.

We also provide an example tutorial in `quick_start.ipynb`.

## Usage

### Command-line Arguments

The `inference.py` script supports several command-line arguments:

| Argument                 | Description                                             | Default                             |
|--------------------------|---------------------------------------------------------|-------------------------------------|
| `--density_map_path`     | Path to the input density map directory (required).     | None                                |
| `--pdb_path`             | Path to the ground truth PDB file (optional).           | None                                |
| `--model_path`           | Path to the pretrained model checkpoint.                | `pretrained_model/checkpoint.pt`    |
| `--output_dir`           | Directory to save the output PDB file.                  | `results`                           |
| `--device`               | Device to run the model on (`cpu` or `cuda`).           | `cuda`                              |
| `--verbose`              | Enable verbose output for debugging.                    | Disabled                            |

### Running the Example

You can run the example directly from the command line:

```bash
python inference.py --density_map_path examples/density_map --pdb_path examples/5uz7.pdb
```

### Using Custom Data

To use E3-CryoFold with your own data, you need to provide a Cryo-EM density map and, optionally, a PDB file for evaluating the predicted structure. For example:

```bash
python inference.py --density_map_path /path/to/your/density_map --pdb_path /path/to/your/ground_truth.pdb --output_dir /path/to/save/results --device cuda
```

## Tutorial

### 1. Preprocessing Density Maps:

To normalize your density maps, run:

	# Normalize you density maps
	$ bash run_data_preparation.bash examples/

After preprocessing, the directory structure should look like:

The organization of the downloaded models should look like:
```text 
E3-CryoFold
├── examples
│   ├── density_map
│   │   ├── map.map
│   │   ├── seq_chain_info.json
│   │   └── normed_map.mrc
|   |── pretrained_model
│   │   ├── checkpoint.pt
```

### 2. Running Inference:

	python inference.py --density_map_path examples/density_map --pdb_path examples/5uz7.pdb 

After inference, the output will be saved in the specified output directory:

```text 
E3-CryoFold
├── results
│   └── output.pdb
```

## References

For a complete description of the method, see:

```
TBD
```

## Contact

Please submit any bug reports, feature requests, or general usage feedback as a github issue or discussion.

- Jue Wang (wangjue@westlake.edu.cn)
- Cheng Tan (tancheng@westlake.edu.cn)
- Zhangyang Gao (gaozhangyang@westlake.edu.cn)

## License

This project is licensed under the MIT License. See the [LICENSE file](https://github.com/A4Bio/E3-CryoFold/blob/main/LICENSE) for details.

