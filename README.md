# Blood Cell Painting DINO (DinoBCP)

Self-supervised DINO architecture modified for [Blood Cell Painting](https://www.biorxiv.org/content/10.1101/2024.05.17.594648v3) images.
Blood Cell Painting is in collaboration with FIMM, Finnish Red Cross Blood Service and FinnGen.
The code builds upon / is inspired by [scDINO](https://github.com/JacobHanimann/scDINO) and [Joona Pohjonen's](https://github.com/jopo666) code, as well as [DINO](https://github.com/facebookresearch/dino) from Facebook.
This is a work in progress, and does not yet include all code required to run it.

## Workflow
Samples from blood donors were extracted as explained in the Blood Cell Painting paper, and imaged by a confocal microscope.

Flattened images were tiled to 540x540, numpy arrays of 4x540x540 created, and the model was taught with them for 13 epochs, with batch size = 64 and vit_small. The main code also uses gradient accumulation and BN for the DINOHead.
Batch effects were removed from the features after training by:
- Scaling to controls and standardizing data to mean = 0 and std = 1
- Applying Harmony for plate and well

For the extracted DINO features, clustering was done by HDBSCAN.
PCA was used to extract final principal components for GWAS analysis, as in Blood Cell Painting.

## Preliminary results

![Training curve](images/training_metrics.png)

![Attention heads](images/attention_heads.png)

![Attention channels](images/attention_channels.png)

![Clustering with HDBSCAN](images/clusters.png)