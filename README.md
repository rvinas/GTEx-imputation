# Deep learning enables fast and accurate imputation of gene expression across tissues [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/rvinas/GAIN-GTEx/blob/master/LICENSE) [![Python 3.5+](https://img.shields.io/badge/python-3.5+-blue.svg)](https://www.python.org/downloads/release/python-350/)
This repository contains the code for the paper [Deep learning enables fast and accurate imputation of gene expression across tissues](https://www.biorxiv.org/content/10.1101/2020.06.09.141689v2.full)

**Authors:** Ramon Viñas, Tiago Azevedo, Eric R. Gamazon, and Pietro Liò

## Abstract
A question of fundamental biological significance is to what extent the expression of a subset of genes can be used to recover the full transcriptome, with important implications for biological discovery and clinical application. To address this challenge, we propose two novel deep learning methods for gene expression imputation. In order to increase the applicability of our approach, we leverage data from GTEx v8, a reference resource that has generated a comprehensive collection of transcriptomes from a diverse set of human tissues. We show that our approaches compare favourably to several standard and state-of-the-art imputation methods in terms of predictive performance and runtime in two case studies and two imputation scenarios. Furthermore, our results indicate strong generalisation on RNA-Seq data from 3 cancer types across varying levels of missingness. Our work can facilitate a cost-effective integration of large-scale RNA biorepositories into genomic studies of disease, with high applicability across diverse tissue types.
## GTEx data
The GTEx gene expression data is available at: https://gtexportal.org/

## Dependencies
```
numpy==1.18.3
tensorflow==2.4.0
tensorflow-probability==0.9.0
scipy==1.4.1
seaborn==0.9.0
pandas==0.24.2
matplotlib==3.0.3
scikit-learn==0.22.1
missingpy==0.2.0
wandb==0.10.15
```

## Training
To train the model run:
```
python imputation.py --config configs/<config_file>.yaml
```

## Architecture

<img src="figures/architecture.png" width="1000">
Figure 1: Architecture of GAIN-GTEx.


## Results
![Tissue results](figures/tissue_results.png)
Figure 2: R2 imputation scores per GTEx tissue with a missing rate of 50%. Each box shows the distribution of the per-gene R2 scores in the extended test set (where all the tissues are equally represented). The colour of each box represents the number of training samples of the corresponding tissue.
![Tissue embeddings](figures/tissue_embeddings.png)
Figure 3: UMAP visualisation of the tissue embeddings from the generator. Colors are assigned toconform to the GTEx Consortium conventions. Note that the central nervous system, consisting ofthe 13 brain tissues, clusters together on the top right corner.
![R2 across missing rate](figures/r2_missingrate.png)
Figure 4: R2 imputation scores per tissue across missing rate for 3 TCGA cancers and their healthy counterpart in GTEx (test set). The shaded area represents one standard deviation of the per-gene R2 scores in the corresponding tissue. The greater the rate of missingness, the lower the performance.
![R2 across missing rate](figures/r2_missingrate.png)
Figure 4: R2 imputation scores per tissue across missing rate for 3 TCGA cancers and their healthy counterpart in GTEx (test set). The shaded area represents one standard deviation of the per-gene R2 scores in the corresponding tissue. The greater the rate of missingness, the lower the performance.
![R2 across missing rate](figures/top_last_30_genes_alzheimer.png) Per-gene imputation $R^2$ scores on genes from the Alzheimer's disease pathway. Each point represents the average $R^2$ score in a tissue type.

## Citation
If you use this code for your research, please cite our paper:

```
@article{Vinas2020gene,
	author = {Vi{\~n}as, Ramon and Azevedo, Tiago and Gamazon, Eric R. and Li{\`o}, Pietro},
	title = {Gene Expression Imputation with Generative Adversarial Imputation Nets},
	year = {2020},
	doi = {10.1101/2020.06.09.141689},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2020/06/13/2020.06.09.141689},
	eprint = {https://www.biorxiv.org/content/early/2020/06/13/2020.06.09.141689.full.pdf},
	journal = {bioRxiv}
}
```


