# AI-Driven Fashion Recommendation Engine 👕👗

## Overview
Standard e-commerce recommendation systems typically rely on either text tags or visual similarity. This project introduces a **Multimodal Deep Learning Recommendation Engine** that simultaneously understands human-written product descriptions and raw product images to predict purchase intent and item popularity.

By combining Natural Language Processing (NLP) with Computer Vision (CV), this system creates a highly personalized, context-aware similarity ranking for fashion items.

## 🧠 Core Architecture
This project is built natively in PyTorch and leverages a custom attention-based fusion pipeline:

1. **Text Feature Extraction (BERT):** Utilizes Hugging Face's `bert-base-uncased` to generate 768-dimensional semantic embeddings from product descriptions.
2. **Visual Feature Extraction (ResNet50):** Uses a pre-trained ResNet50 CNN (with the final classification layer removed) to extract 2048-dimensional visual patterns from 224x224 product images.
3. **Attention-Based Multimodal Fusion:** A custom PyTorch `MultimodalFusionWithAttention` layer that projects both streams into a shared 1024-dimensional space. It applies an attention mechanism to mathematically weigh whether the text or the image is more relevant for a specific item.
4. **Purchase Intent Classifier:** A Multi-Layer Perceptron (MLP) binary classifier utilizing Batch Normalization and Dropout to predict product popularity/purchase intent based on the fused feature profile.

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning Framework:** PyTorch
* **NLP:** Hugging Face Transformers (BERT), NLTK
* **Computer Vision:** Torchvision (ResNet50), OpenCV, PIL
* **Data Processing:** Pandas, NumPy
* **Evaluation Metrics:** Scikit-learn (NDCG, F1-Score, Precision/Recall)

## 📊 Evaluation Metrics
The model's ranking and classification performance was evaluated using:
* **NDCG (Normalized Discounted Cumulative Gain):** To measure the quality of the recommendation ranking.
* **F1-Score / Mean Average Precision:** To validate the accuracy of the purchase intent binary classification.

## 🚀 Setup and Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/fashion-recommendation-engine.git](https://github.com/yourusername/fashion-recommendation-engine.git)
   cd fashion-recommendation-engine