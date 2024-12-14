# Art Classifier Project

This project focuses on using deep learning models to classify artworks based on their **artist** and **style**. The goal is to explore the challenges of fine-grained image classification in the context of art, where overlapping styles and diverse artistic techniques make the task uniquely complex.

## Problem Statement
Art classification is challenging because:
- **Artists often work across multiple styles**, making it hard to associate them with a single set of features.
- **Visual similarities** exist between artworks of the same movement or period, which confuses models.
- **Imbalanced datasets** result in overrepresented artists dominating predictions.

However, solving this problem has important applications in:
- Digital archiving and retrieval of artworks.
- Enhancing user experience in virtual galleries.
- Supporting art historians with stylistic analysis and provenance studies.

## Methodology
### Models Used
1. **VGG19 and Inception**
   - Pre-trained on ImageNet for transfer learning.
   - Fine-tuned with custom classification heads for predicting artist and style.

2. **Vision Transformers (ViTs)**
   - Investigated for their ability to capture global patterns in images by treating them as sequences of patches.

### Techniques and Tools
- **Dataset Preprocessing**
  - Merging datasets from multiple sources.
  - Addressing imbalances through data augmentation (flipping, rotating, color jittering).

- **Training Setup**
  - Loss Function: Cross-entropy loss for classification.
  - Optimizer: Adam optimizer with learning rate scheduling and weight decay.

- **Experimental Features**
  - Exploring style predictions as auxiliary inputs to improve artist classification.
  - Custom evaluation metrics for hierarchical error analysis (e.g., style-level confusion vs. artist-level confusion).

## Challenges
1. **Artist Overlap Across Styles**
   - Artists like Picasso work across movements, leading to overlapping features.
2. **Imbalanced Dataset**
   - Some artists have far more images than others, biasing predictions.
3. **Computational Complexity**
   - Training on high-resolution images with deep models requires significant resources.
4. **Low Validation Accuracy With 49 Classes**
   - Self-trained models predicting artists over 40 lables currently achieve validation accuracy of ~10–11%, close to random guessing, indicating the complexity of the task.

5. **Higher Accuracy With 15 Classes*
   - Pretrained models currently achieve validation accuracy of ~89%.

## Key Findings
- **Style is easier to predict than artist.**
  - Style predictions may provide useful context for artist classification.
- **Inception outperforms VGG19 slightly** due to its multi-scale feature extraction.
- **Vision Transformers show promise** in handling complex patterns but require more tuning.
- **Data augmentation stabilizes training** and improves model generalization.
- **Class imbalance impacts results** significantly, highlighting the need for weighted loss or oversampling techniques.

## Results
### Current Performance
- Training loss stabilizes with sufficient data augmentation.
- Validation accuracy is ~10–11% for self trained models, however, 89% for pretrained models e.g. ViT.

### Future Directions
- Incorporate **style embeddings** as auxiliary inputs to improve artist classification.
- Experiment with **weighted loss functions** or oversampling to address data imbalance.
- Optimize architectures or use model pruning for computational efficiency.
- Expand the dataset to include more artists and styles for better generalization.

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/ychenhq/Classical-Art-Classifier.git
   ```
2. Prepare the dataset:
   - Ensure datasets are organized into folders by artist and style.
   - Download the dataset file from
     https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time

## Acknowledgments
- Pre-trained models (VGG19, Inception) were sourced from TensorFlow/Keras.
- Dataset contributions from online art archives and Kaggle repositories.

---
This project highlights the fascinating intersection of AI and art, addressing unique challenges in a creative domain. Contributions and suggestions for improvement are welcome!
