
# Vehicle Orientation Classification Using Deep Learning

## Project Description

This project was developed as part of a Computer Vision semester course. The objective is to classify vehicle orientation (front or back) in images using deep learning techniques. The model leverages Convolutional Neural Networks (CNNs) to efficiently extract features and provide accurate classification results. This project highlights the application of advanced computer vision techniques for real-world image classification tasks.

----------

## 🚀 How to Run the Project

### Step 1: Clone the Repository

To start, clone this repository to your local machine:

```bash
git clone https://github.com/your-username/vehicle-orientation-classification.git
cd vehicle-orientation-classification

```

### Step 2: Set Up a Virtual Environment

Create and activate a Python virtual environment to manage dependencies:

-   **For Linux/Mac**:
    
    ```bash
    python -m venv venv
    source venv/bin/activate
    
    ```
    
-   **For Windows**:
    
    ```bash
    python -m venv venv
    venv\Scripts\activate
    
    ```
    

### Step 3: Install Dependencies

Install the required Python libraries from the `requirements.txt` file:

```bash
pip install -r requirements.txt

```

### Step 4: Run the Jupyter Notebook

Launch the Jupyter Notebook interface to train or evaluate the model:

```bash
jupyter notebook notebook.ipynb

```

### Step 5: Run the Python Script (Optional)

Alternatively, execute the Python script directly to perform the same operations:

```bash
python script.py

```

----------

## 📊 Dataset

### Dataset Source

The dataset is sourced from [PakWheels](https://www.pakwheels.com/), containing images of vehicle fronts and backs.

### Dataset 

The dataset is organized into the following structure:

```
car_dataset/
├── car_back/     # Images of vehicle back view
├── car_front/    # Images of vehicle front view

split_dataset/
├── train/
│   ├── car_back/
│   └── car_front/
├── test/
│   ├── car_back/
│   └── car_front/
└── val/
    ├── car_back/
    └── car_front/

```

-   **Raw Dataset**: Located in the `car_dataset/` directory.
-   **Split Dataset**: Divided into training, validation, and testing subsets in the `split_dataset/` directory.

----------

## Model Details

### Feature Extraction with VGG and SVM Classifier

we used **VGG** for feature extraction and then applied the **Support Vector Machine (SVM)** algorithm for classification. The results were significantly improved.

Results indicate that the SVM classifier, when combined with the features extracted by VGG, achieved excellent performance across the training, validation, and test sets. The high training accuracy, precision, recall, and F1 scores suggest that the model is well-optimized and capable of distinguishing between the "front" and "back" categories with minimal error. Notably, the slight decrease in recall and F1 score on the test set (compared to training) suggests a small drop in performance on unseen data, which is common in machine learning tasks.

----------

## 📈 Results

Model (VGG and SVM Classifier)  the following performance metrics:

-   **Accuracy**: 95%
-   **Precision**: 96%
-   **Recall**: 94%
-   **F1-Score**: 95%
-   **Cross-validation Scores**: 0.98412698, 0.91269841, 0.98412698, 0.95238095, 0.984

These results demonstrate the model's effectiveness in classifying vehicle orientation under varying conditions.

----------

## References

-   [PakWheels](https://www.pakwheels.com/) - Source of vehicle images.


