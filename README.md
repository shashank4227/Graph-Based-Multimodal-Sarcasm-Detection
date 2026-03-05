# Multimodal Sarcasm Detection using Graph Attention Networks

This repository contains the implementation of a Multimodal Sarcasm Detection system. The system leverages multiple modalities—Text, Audio, Video, and Context—and models their interactions using a Graph Attention Network (GATv2) in PyTorch Geometric.

## Repository Overview

*   **`Feature_Extraction.ipynb`**: Handles the extraction of features from raw data.
    *   **Text Features**: Extracted using the pre-trained `roberta-base` model.
    *   **Video Features**: Frame-level features extracted using a pre-trained `ResNet50` model.
    *   **Audio Features**: Acoustic features like MFCCs, pitch, and spectral centroid extracted using `librosa`.
    *   **Context**: Concatenates contextual text and visual features.
    *   Saves the extracted features as `.npy` arrays for fast loading during training.
*   **`MultiModal Sarcasm Detection.ipynb`**: Contains the core Graph Neural Network (GNN) implementation and training pipeline.
    *   Constructs a graph for each sample where nodes represent different modalities (Text, Audio, Video, Context).
    *   Defines inter-modality edges (e.g., a modality triangle for Text-Audio-Video and a context star connecting Context to the others).
    *   Implements the `SarcasmGATv2` model, utilizing Graph Attention layers (`GATv2Conv`) to allow modalities to attend to one another.
    *   Trains the network utilizing an AdamW optimizer, Cosine Annealing scheduler, and evaluates using metrics including Accuracy, F1-Score, and AUC.

## Requirements

The notebooks are designed to run in environments like Google Colab, but can be run locally with the following primary dependencies:

*   Python 3.x
*   PyTorch
*   PyTorch Geometric (`torch_geometric`, `pyg_lib`, `torch_scatter`, `torch_sparse`, `torch_cluster`, `torch_spline_conv`)
*   Transformers (`transformers`)
*   Torchvision (`torchvision`)
*   OpenCV (`opencv-python`)
*   Librosa (`librosa`)
*   MoviePy (`moviepy`)
*   Scikit-learn, NumPy, Matplotlib, tqdm

*(Note: The notebooks include cells to install the necessary libraries via `pip`)*

## Usage

1.  **Data Preparation & Feature Extraction**:
    *   Ensure your raw dataset (`sarcasm_data.json` and associated utterance/context videos) is placed in your configured directory.
    *   Run `Feature_Extraction.ipynb` to process the raw multimodal data and generate `_features.npy` arrays.
2.  **Model Training**:
    *   Open `MultiModal Sarcasm Detection.ipynb`.
    *   Update the `BASE_PATH` in the configuration cell to point to the directory containing your extracted features.
    *   Run the notebook to instantiate the dataset, construct graphs, train the `SarcasmGATv2` model, and evaluate its performance on the test set.

## Model Architecture

![Model Architecture](Model Architecture.png)

The overall architecture of the proposed multimodal sarcasm detection system is shown above. The model integrates **text, audio, visual, and contextual information** to capture complementary cues required for sarcasm understanding. Textual utterances are encoded using the pre-trained **RoBERTa** model to obtain semantic embeddings. Audio signals are processed to extract acoustic features such as **MFCCs, pitch, and spectral characteristics**, while visual frames are encoded using a **ResNet50** backbone to capture facial expressions and scene-level information. Contextual information is derived by combining textual and visual signals from surrounding dialogue to provide additional conversational cues.

The extracted modality features are projected into a **shared latent representation space** and represented as nodes in a **heterogeneous graph**. Edges are defined between modalities to model their relationships, including connections between **Text–Audio–Video** modalities and additional links from the **Context** node to each modality. The graph is processed using **GATv2Conv layers** from PyTorch Geometric, allowing each modality to attend to others through learnable attention weights. The resulting node representations are aggregated using **global mean pooling** and passed through a **multilayer perceptron (MLP)** classifier to predict the probability of sarcasm.

## Model Architecture Highlights

*   **Feature Projections**: Initial Linear + LayerNorm + GELU layers to project extracted features from disparate dimensionalities into a common hidden dimension space.
*   **Graph Attention**: Two `GATv2Conv` layers to perform node feature aggregation across modalities based on learned attention weights.
*   **Pooling & Classification**: `global_mean_pool` across nodes followed by an MLP layer predicting the binary sarcasm label.
