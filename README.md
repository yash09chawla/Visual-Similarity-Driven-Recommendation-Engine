# Visual Similarity Driven Recommendation Engine

**Kaggle Dataset Link:** [Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)

## Overview

A deep learning-based recommendation system that finds visually similar fashion products using image embeddings. The system leverages transfer learning with ResNet50 to extract visual features and uses nearest neighbor search to recommend similar items based on visual similarity rather than metadata or user behavior.

## Features

- 🖼️ **Visual Feature Extraction**: Uses pre-trained ResNet50 model to extract high-dimensional feature vectors from product images
- 🔍 **Similarity Search**: Implements k-nearest neighbors algorithm with Euclidean distance metric
- 🌐 **Interactive Web Interface**: Streamlit-based web application for easy image upload and recommendation visualization
- ⚡ **Efficient Processing**: Pre-computed embeddings for fast real-time recommendations

## Technology Stack

- **Deep Learning**: TensorFlow/Keras with ResNet50 (ImageNet weights)
- **Machine Learning**: scikit-learn (NearestNeighbors)
- **Web Framework**: Streamlit
- **Image Processing**: PIL, OpenCV
- **Data Storage**: Pickle for embedding serialization

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd GC
```

2. Install required dependencies:
```bash
pip install tensorflow streamlit pillow scikit-learn numpy opencv-python tqdm
```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) and extract images to an `images/` folder

## Usage

### Step 1: Generate Embeddings

First, extract features from all product images:

```bash
python app.py
```

This will:
- Process all images in the `images/` directory
- Extract features using ResNet50
- Save embeddings to `embeddings.pkl` and filenames to `filenames.pkl`

### Step 2: Run the Web Application

Launch the Streamlit app:

```bash
streamlit run main.py
```

Upload an image through the web interface to get 5 visually similar product recommendations.

### Step 3: Test Locally (Optional)

Test the recommendation system with a sample image:

```bash
python test.py
```

## Project Structure

```
Visual-Similarity-Driven-Recommendation-Engine/
├── app.py              # Feature extraction script
├── main.py             # Streamlit web application
└── test.py             # Local testing script
```

## How It Works

1. **Feature Extraction**: ResNet50 (pre-trained on ImageNet) extracts visual features from images, which are then normalized to create 2048-dimensional embedding vectors
2. **Similarity Computation**: Uses k-nearest neighbors with Euclidean distance to find the most similar products
3. **Recommendation**: Returns the top 5 visually similar items based on the extracted features


## License

This project is open source and available for educational purposes.
