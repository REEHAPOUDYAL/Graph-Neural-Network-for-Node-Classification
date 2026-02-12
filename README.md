# Graph Convolutional Network (GCN) for Node Classification - Wiki-CS Dataset

## Overview
This project implements a **Graph Convolutional Network (GCN)** to perform **node classification** on the **Wiki-CS dataset**, which contains Wikipedia computer science articles connected through hyperlinks.

The goal is to learn meaningful node embeddings using **graph-based message passing** and classify each node into one of **10 categories**.

---

## Dataset (Wiki-CS)
- **Nodes:** 11,701 Wikipedia CS articles  
- **Edges:** 297,291 hyperlinks between articles  
- **Node Features:** 300-dimensional embeddings  
- **Classes:** 10 categories  
- **Task:** Node classification using train/val/test masks  

---

## Model Architecture
A deep **3-layer GCN** was implemented using sequential `GCNConv` layers.

### Key Components
- **Input Features:** 300 dimensions  
- **Hidden Layer:** 64-dimensional latent embeddings  
- **Output Layer:** 10-class logits  

### Techniques Used
- **ReLU Activation** after each GCN layer
- **Dropout (p = 0.4)** between layers for regularization
- **Skip Connection** in the second layer to improve gradient flow
- **Log-Softmax** output for multi-class classification

---

## Training Setup
- **Optimizer:** Adam  
- **Learning Rate:** 0.01  
- **Loss Function:** Negative Log Likelihood Loss (NLLLoss)  
- **Weight Decay:** L2 regularization to reduce overfitting  
- **Early Stopping:** Patience-based early stopping using validation performance  
- **Checkpointing:** Best model saved automatically based on validation loss  

---

## Evaluation
Performance was measured using accuracy on:
- Training Mask
- Validation Mask
- Test Mask

Final training results:

- **Epoch:** 40  
- **Loss:** 0.4088  
- **Train Accuracy:** 0.8054  
- **Validation Accuracy:** 0.8060  
- **Test Accuracy:** 0.7913  

---

## Results
The trained GCN achieved a **Test Accuracy of 79.13%**, showing strong generalization on unseen Wiki-CS nodes.

To analyze embedding quality, **t-SNE visualization** was generated, showing clustering of node embeddings across the 10 classes.

---

## Conclusion
This project demonstrates how Graph Neural Networks can effectively learn from **non-Euclidean structured data** such as graphs.  
The GCN model successfully captured relational dependencies in the Wiki-CS hyperlink network and produced accurate class predictions through neighborhood aggregation and feature transformation.

---

## Technologies Used
- Python
- PyTorch
- PyTorch Geometric (GCNConv)
- Matplotlib (visualization)
- t-SNE (embedding visualization)

---
