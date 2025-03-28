
# MTMNet: Multi-task Learning with Limited Data for Source-free Cross-domain Few-shot Semantic Segmentation

## Abstract

Cross-domain few-shot semantic segmentation (CD-FSS) aims to achieve efficient image segmentation within a target domain using limited annotated data. Existing methods often require access to source data during training. However, with growing concerns over data privacy and the need to reduce data transfer and training costs, developing a CD-FSS solution that does not require access to source data is necessary. 

To address this issue, this paper proposes a multi-task learning-based source-free cross-domain few-shot semantic segmentation method (MTLNet). This method optimizes and fine-tunes the model using a minimal amount of target domain data without accessing source domain data, thereby improving segmentation performance.  

Key contributions include:
- **Hierarchical Mask module**: Combines contrastive learning and supervised learning to prevent model overfitting and enhance generalization.
- **Multi-task loss function**: Helps capture general patterns and reduces overfitting to a single task, improving robustness in new data.
- **Triplet Contextual Alignment strategy**: Manages dependencies between tasks, optimizing performance in semantic segmentation.

Experimental results demonstrate that MTLNet significantly outperforms the CD-FSS baseline, showcasing the potential of multi-task learning when trained on extremely small amounts of target domain data.



## Datasets

[](https://github.com/slei109/PATNet?tab=readme-ov-file#datasets)

The following datasets are used for evaluation in CD-FSS:

### Target domains:

[](https://github.com/slei109/PATNet?tab=readme-ov-file#target-domains)

-   **Deepglobe**:
    
    Home:  [http://deepglobe.org/](http://deepglobe.org/)
    
    Direct:  [https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset)
    
    Preprocessed Data:  [https://drive.google.com/file/d/10qsi1NRyFKFyoIq1gAKDab6xkbE0Vc74/view?usp=sharing](https://drive.google.com/file/d/10qsi1NRyFKFyoIq1gAKDab6xkbE0Vc74/view?usp=sharing)
    
-   **ISIC2018**:
    
    Home:  [http://challenge2018.isic-archive.com](http://challenge2018.isic-archive.com/)
    
    Direct (must login):  [https://challenge.isic-archive.com/data#2018](https://challenge.isic-archive.com/data#2018)

    
-   **Chest X-ray**:
    
    Home:  [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/)
    
    Direct:  [https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels)

-   **FSS-1000**:
    
    Home:  [https://github.com/HKUSTCV/FSS-1000](https://github.com/HKUSTCV/FSS-1000)
    
    Direct:  [https://drive.google.com/file/d/16TgqOeI_0P41Eh3jWQlxlRXG9KIqtMgI/view](https://drive.google.com/file/d/16TgqOeI_0P41Eh3jWQlxlRXG9KIqtMgI/view)
    
-   **WBC**:
    
    Home:  [https://github.com/zxaoyou/segmentation_WBC](https://github.com/zxaoyou/segmentation_WBC)
    
-   **SegPC-2021**:
    
    Home:  [https://www.kaggle.com/datasets/sbilab/segpc2021dataset](https://www.kaggle.com/datasets/sbilab/segpc2021dataset)
    
    
