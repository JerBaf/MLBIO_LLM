# MLBIO_LLM
This repo contains the code developped during my project in the MLBio lab at EPFL.

## Abstract

Large Language Models (LLMs) are more and more used outside the NLP field due to their remarkable performances. Their ability to capture complex interactions between elements is key for many field, and biology makes no exception. Gene to gene interactions have been studied for many years and their understanding is crucial for a broad range of downstream task such as disease causal gene or drug target detection. In this project, we tried to assess if we could use prior knowledge to extend the generalization capabilities of LLM to unseen genes and tissues during training. Our results showed that we were not successful in designing a proper scheme to leverage this information. Nevertheless this project revealed interesting future development paths for the next generation of LLM applied to single cell data. 

## Structure
```
│   LLM_Project_Report.pdf
│   README.md
│
├───img
│       CellLM.png
│       CellLM_performances.png
│       LM_performances.png
│       Small_LM_performances.png
│       MLP_performances.png
│       Cortex_histogram.png
│       Hippocampus_histogram.png
│
└───src
    │   train.py
    │   data.py
    │   bert.py
    │   modeling_bert.py
    │   configuration_bert.py
    │
    ├───notebooks
    │   ├───graph_creation
    │   │       cell2loc_go.ipynb
    │   │       gears_go.ipynb
    │   │
    │   ├───results_visualization
    │   │       cell2loc_results.ipynb
    │   │
    │   └───data_processing
    │           data_processing.ipynb
    │           hic.ipynb
    │           cell2loc_gene_selection.ipynb
    │   
```

In the `img` folder, you find the main graphics used in the report.

In the `src` folder, you find the files containing the code for the project. You have first the different .py files which contain all the code and pipeline required to reproduce the results and continue the project. Note that these files are complementary to the code from Stéphane D'Ascoli on the Cell2loc project. Thus they will not work as standalone modules. You find these different files:

- `train.py`: Training pipeline for the MERFISH data.
- `data.py`: Data related functions for the MERFISH data.
- `bert.py`: BERT modeling for working with single cell data.
- `modeling_bert.py`: Adaptation of the core module in the HuggingFace repo for BERT implementation.
- `configuration_bert.py`: Configuration module in the HuggingFace repo for BERT implementation.


Furthermore, in the `src` folder, you have a `notebook` folder which contains the different notebooks used to build the project. They are here for consistency and to allow possible debugging. The notebook folder is split in different subfolders assigned to specific tasks:
- `graph_creation`: The folder contains the notebooks used to convert raw data into graph representation. 
    - `cell2loc_go.ipynb`: Create the prior knowledge graphs to work with the cell2loc project (MERFISH data). 
    - `gears_go.ipynb`: Create the prior knowledge graphs to work with the CellLM project, mostly using GEARS prior knowledge graph (scRNA data).
- `results_visualization`: Contain the file `cell2loc_results.ipynb` which allow to visualize the training and test curves for the cell2loc project.
- `data_processing`: The folder contains the notebooks used to process the data. 
    - `data_processing.ipynb`: Process the Mouse Cell Atlas data to work with CellLM. 
    - `hic.ipynb`: Process the HiC data to create the adjacency matrix of the prior knowledge graph.
    - `cell2loc_gene_selection.ipynb`: Select the genes for the cell2loc project that are the most differentiated accross cell types.


## Get Data

To reproduce the results it is advised to use the [runAI infrastucture](epfl.run.ai/) and the MLBio cluster to gain access to the data. Once logged, you can use the ssh to get the main files to reproduce the results. To do so, follow these instructions: 

- Connect to the MLBio cluster
- Go to the directory: `\mlbiodata1\baffou\final_files`
- Copy all the files to the wanted directory

Wait until the end of download and you will retrieve all files needed for computation, including the ones in this repository.

## RunAI
RunAI is the platform used by the IC clusters at EPFL to make large scale ML. Here is a guide to use it properly
### Installation
As long as the cluster is under runAI 2.8 there is a distinction between windows and linux/mac installations. But the migration is in progress, so please check before with the IC team if the cluster has been updated. If so, you can just follow the linux/mac instruction for any distribution you are using.
#### Linux/Mac
The first thing you need is the CLI for kubernetes: https://kubernetes.io/docs/tasks/tools/
Then you can download runAI CLI by going to epfl.run.ai and in the upper right corner, click on the question mark → Research Command Line Interface.
And that’s it, you can (almost) use runAI!
#### Windows
For windows things are a little more complicated but you can follow these instructions and they are quite clear: https://docs.run.ai/admin/researcher-setup/cli-install/#install-runai-cli
To sum up the steps you need to:
Download docker desktop
Download the kubernetes config file associated with the epfl runAI cluster
Download the docker image and replace the config file
Adapt the URL of the cluster in the image by the one provided on https://epfl.run.ai/clusters
Build the image using the provided build.sh script

And that’s it, you can (almost) use runAI!
### Ask for PVCs
There are several ways to transfer code and data to the different pods but the simplest (and safest) way is to bind the pod with existing MLBio infrastructure that you can access. Namely the /mlbiodata1 and /scratch storage facilities. Note that the /scratch is different from the /scratch on the MLBio cluster. You need to ask the IT team to give you access to these two different volumes using PVCs (private volume claim). These PVCs are already in place, you just need the access.
In practice mlbiodata1 will be useful for data, environment and code storage. And scratch will be useful to save results and checkpoints. It is super important to save checkpoints as the lifetime of pods is limited and they can be canceled by the job scheduler.
### List of command to get started
#### Create Docker Image (only for windows)
docker run -it runai-cli bash
#### Log in to runai
runai login
runai config project mlbio-baffou
runai list
#### Install Kubernetes (only for windows)
apt-get update
apt-get install -y ca-certificates curl
curl -fsSLo /etc/apt/keyrings/kubernetes-archive-keyring.gpg https://packages.cloud.google.com/apt/doc/apt-key.gpg
curl -fsSLo /etc/apt/keyrings/kubernetes-archive-keyring.gpg https://dl.k8s.io/apt/doc/apt-key.gpg
echo "deb [signed-by=/etc/apt/keyrings/kubernetes-archive-keyring.gpg] https://apt.kubernetes.io/ kubernetes-xenial main" | tee /etc/apt/sources.list.d/kubernetes.list
apt-get update
apt-get install -y kubectl
#### Create interactive session
runai submit \
  --name sandbox \
  --interactive \
  --gpu 1 \
  --image ic-registry.epfl.ch/mlo/pytorch \
  --pvc runai-mlbio-baffou-scratch:/scratch \
  --pvc runai-mlbio-baffou-data1:/data1 \
  --large-shm --host-ipc \
  --command -- sleep infinity

Or if you do no want a GPU: 

runai submit \
  --name sandbox \
  --interactive \
  --image ubuntu:latest \
  --pvc runai-mlbio-baffou-scratch:/scratch \
  --pvc runai-mlbio-baffou-data1:/data1 \
   --command -- sleep infinity

#### Enter session
runai exec -it sandbox /bin/bash
#### Set up conda env
conda init bash
(exit and enter pod)
conda activate path_to_cell2loc_bin_env

#### Launch Training

python train.py --loss classification --positional_encodings relative_key_query --adjacency_path adjacency_p2p_and_hic_matrix.npz --selected_genes_path selected_gene_names.json --bin_num 7

#### Stop session
runai delete job sandbox
#### Copy files to local (windows)
mkdir /scratch
kubectl cp sandbox-0-0:/scratch/baffou/experiments/_/train.log /scratch/_.log
docker cp container:/scratch/_.log M:\scratch\jeremy\runs\cell2loc\runai\train_logs\_.log

## Side Notes

### Data Paths

Many of the datapaths for training, runAI or custom script requires hardcoding. Please check that you correctly adapt the paths especially in the training script and in the runAI pipelines.

### Requirements

All the code is under python 3.8.16. To have the correct environment you can use the one in the folder from the data repo precised in the previous section.
