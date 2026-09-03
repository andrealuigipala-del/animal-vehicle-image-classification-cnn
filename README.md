# Animal Vehicle Classification with CNN

## Overview

This project explores the development of a **Computer Vision system for automatic image classification**, designed to distinguish between **animals and vehicles** in an urban road environment.

The project was developed for **VisionTech Solutions**, with the goal of investigating how deep learning can support automated wildlife monitoring and road-safety systems. In a potential real-world scenario, cameras installed along urban roads could use image classification to identify animals entering the roadway and trigger appropriate safety responses.

The solution is based on **Convolutional Neural Networks (CNNs)** implemented with **TensorFlow and Keras**, and trained on the **CIFAR-10 dataset**.

Rather than performing the original 10-class CIFAR-10 classification task, the dataset was adapted to the specific business problem by grouping the relevant classes into two categories:

- **Animal:** bird, cat, deer, dog, frog, horse
- **Vehicle:** automobile, truck

Airplane and ship images were excluded because they are not relevant to the terrestrial road-safety scenario.

The resulting task is therefore formulated as a **binary image classification problem**:

- `0` → Animal
- `1` → Vehicle

---

## Business Objective

VisionTech Solutions aims to develop an automated image recognition system that could support urban road safety and wildlife monitoring.

A system of this type could potentially:

- Automatically identify animals and vehicles from camera images
- Reduce the need for manual monitoring
- Support real-time road surveillance
- Help detect animals entering roadways
- Reduce the risk of animal-vehicle collisions
- Provide information that could support electronic warning systems

The project represents a **proof of concept** for this type of application.

---

## Approach

The project follows an end-to-end deep learning workflow:

**CIFAR-10 → Dataset Filtering → Binary Labeling → Class Balancing → Normalization → Data Augmentation → CNN Training → Evaluation → Error Analysis**

The original CIFAR-10 dataset contains 10 classes.

For this project, only the classes relevant to the target application were retained.

### Selected Classes

**Animals**

- Bird
- Cat
- Deer
- Dog
- Frog
- Horse

**Vehicles**

- Automobile
- Truck

The following classes were removed:

- Airplane
- Ship

This transformation converts the original multiclass problem into a binary classification problem.

---

## Dataset Preparation

The CIFAR-10 dataset is loaded directly using the TensorFlow/Keras API.

The original labels are filtered to retain only the classes required for the application.

The labels are then converted into binary targets:

- `0` → Animal
- `1` → Vehicle

The images maintain the original CIFAR-10 dimensions:

**32 × 32 × 3**

where the three channels represent RGB color information.

### Pixel Normalization

Image pixel values originally range from `0` to `255`.

They are normalized to the `[0, 1]` range by dividing the image arrays by `255`.

This preprocessing step improves the numerical stability of the neural network during training.

---

## Class Imbalance

After grouping the CIFAR-10 classes, the resulting dataset is significantly imbalanced:

- **30,000 animal images**
- **10,000 vehicle images**

This imbalance could cause the model to favor the majority class.

To address this issue, the animal class was randomly downsampled to match the number of vehicle samples.

The resulting training dataset contains:

- **10,000 animal images**
- **10,000 vehicle images**

A fixed random seed (`42`) is used to make the sampling process reproducible.

The test set is kept separate and is not downsampled.

---

## Data Augmentation

Because downsampling reduces the amount of training data available from the animal class, **data augmentation** was introduced to increase the variability of the training samples.

The augmentation pipeline uses Keras `ImageDataGenerator` with:

- Random rotations up to 10 degrees
- Width shifts
- Height shifts
- Zoom
- Horizontal flipping

These transformations are applied dynamically during training.

The objective is to expose the CNN to slightly different versions of the training images and improve its ability to generalize.

---

## CNN Models

Two different CNN architectures were implemented and compared.

The purpose of this comparison was to investigate the relationship between:

- Model complexity
- Training time
- Representation capacity
- Classification performance

---

## CNN 1 — Lightweight Baseline

The first architecture was intentionally designed as a relatively lightweight CNN.

It contains:

- Input layer
- One convolutional layer
- Max Pooling
- Flatten layer
- Dense output layer

The convolutional layer uses:

- **96 filters**
- **3 × 3 kernel**
- **ReLU activation**

The output layer uses a single neuron with **Sigmoid activation**, which is appropriate for binary classification.

The model is compiled using:

- **Adam optimizer**
- **Binary Cross-Entropy loss**
- **Accuracy metric**

This model acts as a baseline and provides a useful reference for evaluating the benefits of increasing network depth.

---

## CNN 2 — Deeper Architecture

The second architecture increases the capacity of the network through multiple convolutional blocks.

The model uses progressively larger numbers of filters:

**32 → 64 → 128**

The architecture includes:

- Multiple `Conv2D` layers
- `MaxPool2D`
- `BatchNormalization`
- `Dropout`
- Fully connected layer
- Sigmoid output layer

The deeper architecture is designed to learn hierarchical visual representations.

Earlier convolutional layers can learn low-level patterns such as:

- Edges
- Colors
- Simple textures

Deeper layers can learn increasingly complex visual structures associated with animals and vehicles.

### Batch Normalization

`BatchNormalization` is used to stabilize the training process and improve the optimization of the deeper network.

### Dropout

`Dropout` is used as a regularization technique to reduce the risk of overfitting.

---

## Training

Both models are trained using:

- **Adam optimizer**
- **Binary Cross-Entropy loss**
- **Accuracy metric**
- **Batch size: 64**
- **10 epochs**

The balanced training dataset is passed through the data augmentation pipeline during training.

Training and validation metrics are recorded through the Keras `History` object.

The training history is exported to JSON files, making it possible to preserve the evolution of:

- Training accuracy
- Validation accuracy
- Training loss
- Validation loss

---

## Model Persistence

The implementation includes functionality for saving and loading trained model weights.

The CNN weights are stored using the Keras `.weights.h5` format.

Training histories are stored as JSON files.

The notebook also includes optional functionality for downloading previously trained weights from Google Drive using `gdown`.

This allows the computationally expensive training stage to be skipped when previously trained weights are available.

---

## Evaluation

The models are evaluated using several metrics and visualizations.

The analysis includes:

- Accuracy
- Precision
- Recall
- Binary Cross-Entropy loss
- Confusion Matrix
- Prediction probability distributions
- Training and validation curves
- Classification threshold analysis

Using multiple evaluation methods provides a more complete understanding of model behavior.

---

## Confusion Matrix

A confusion matrix is generated to analyze the types of errors produced by the CNN.

The matrix allows the following cases to be identified:

- Correctly classified animals
- Correctly classified vehicles
- Animals incorrectly classified as vehicles
- Vehicles incorrectly classified as animals

This analysis is particularly relevant for the intended road-safety application because different types of classification errors may have different operational consequences.

---

## Prediction Probability Analysis

The CNN uses a sigmoid output neuron to generate a probability associated with the vehicle class.

The predicted probabilities are visualized through a histogram.

This makes it possible to investigate how strongly the model separates the two classes and whether predictions tend to cluster close to the extremes or around the decision boundary.

The deeper CNN produces a more clearly defined probability distribution, suggesting more confident predictions compared with the simpler architecture.

---

## Classification Threshold Analysis

The standard classification threshold is:

`0.5`

Predictions above this value are classified as:

**Vehicle**

Predictions below this value are classified as:

**Animal**

The project also evaluates alternative thresholds:

- `0.3`
- `0.5`
- `0.7`
- `0.9`

For each threshold, precision and recall are calculated.

This experiment demonstrates how changing the classification threshold affects the balance between different types of classification errors.

In a real-world application, the optimal threshold would depend on the operational objective and the relative cost of false positives and false negatives.

---

## Model Comparison

The two CNN architectures demonstrate a clear trade-off between computational cost and model complexity.

### CNN 1

The lightweight CNN:

- Has a simpler architecture
- Requires less computational effort
- Has a significantly shorter training time
- Provides a useful baseline
- Shows a more aggressive behavior toward the vehicle class

The training time was approximately **12 minutes** in the experimental environment.

### CNN 2

The deeper CNN:

- Uses multiple convolutional layers
- Has a higher representation capacity
- Uses Batch Normalization
- Uses Dropout
- Produces more balanced classification behavior
- Requires significantly more computational resources

The training time was approximately **44 minutes** in the experimental environment.

The deeper architecture therefore provides a more expressive model at the cost of substantially increased training time.

---

## Key Insights

### 1. Dataset Adaptation Is Important

The original CIFAR-10 dataset was not directly aligned with the business problem.

The dataset therefore had to be adapted by:

- Removing irrelevant classes
- Grouping multiple classes
- Converting the target into a binary classification problem
- Addressing class imbalance

This demonstrates the importance of translating a business problem into an appropriate machine learning formulation.

### 2. Class Imbalance Can Affect Model Behavior

The initial 3:1 ratio between animals and vehicles could lead to a biased classifier.

Balancing the training data provides a more appropriate basis for learning the two target classes.

### 3. Data Augmentation Helps Compensate for Downsampling

Downsampling removes a substantial number of animal samples.

Data augmentation partially compensates for this reduction by exposing the model to different variations of the available images during training.

### 4. Deeper Networks Can Learn More Complex Representations

The second CNN contains multiple convolutional blocks and therefore has greater capacity to learn hierarchical image representations.

However, greater capacity also comes with higher computational cost.

### 5. Accuracy Alone Is Not Enough

Accuracy provides an overall measure of classification performance, but it does not explain which classes are being misclassified.

Confusion matrices, precision, recall and probability distributions provide additional information that is particularly important for a safety-oriented application.

### 6. The Decision Threshold Is an Important Parameter

The classification threshold can be adjusted according to the desired operational behavior.

This creates a connection between the statistical output of the model and the requirements of the final application.

---

## Limitations

This project should be considered a **proof of concept**, rather than a production-ready autonomous driving system.

The most significant limitation is the use of CIFAR-10.

CIFAR-10 contains very small images with a resolution of only:

**32 × 32 pixels**

This is substantially different from real-world road-camera imagery.

A production system would need to handle:

- Higher image resolutions
- Night-time conditions
- Different weather conditions
- Motion blur
- Occlusion
- Different camera perspectives
- Objects at different distances
- Complex urban backgrounds
- Multiple objects in the same image
- Partial visibility of animals or vehicles

Another important limitation is that the current system performs **image-level classification**.

It determines whether an image belongs to the animal or vehicle category, but it does not identify the exact location of an object inside the image.

For an autonomous driving application, an **object detection** approach would therefore be more appropriate.

---

## Future Development

Possible future improvements include:

- Training on higher-resolution, domain-specific road datasets
- Transfer learning with pretrained CNN architectures
- Evaluation of architectures such as ResNet or EfficientNet
- Object detection instead of image-level classification
- Real-time inference
- Model quantization
- Inference optimization
- Confidence calibration
- Automated threshold optimization
- Testing under different environmental conditions
- Integration with real-time camera streams

A particularly important next step would be replacing CIFAR-10 with a dataset specifically collected from **urban road environments**.

This would allow the model to learn from images that better represent the conditions encountered during actual deployment.

---

## Technologies

- **Python**
- **TensorFlow**
- **Keras**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**
- **gdown**
- **Google Colab**

---

## Installation & Usage

### 1. Clone the Repository

Clone the repository and move into the project directory.

    git clone <repository-url>
    cd animal-vehicle-classification-cnn

### 2. Create a Virtual Environment

It is recommended to use a virtual environment to isolate the project dependencies.

    python -m venv venv

Activate the virtual environment.

**Windows:**

    venv\Scripts\activate

**macOS / Linux:**

    source venv/bin/activate

### 3. Install Dependencies

Install the required Python libraries using the provided `requirements.txt` file:

    pip install -r requirements.txt

The `requirements.txt` file contains the dependencies required by the project.

### 4. Run the Python Implementation

The main Python implementation is located in the `main/` directory.

    python main/<python-file>.py

### 5. Run the Notebook

The complete experimental workflow is available in the `notebook/` directory.

The notebook can be opened using:

- Google Colab
- Jupyter Notebook

If running locally with Jupyter:

    jupyter notebook

Then open the notebook contained in:

    notebook/

The notebook includes the complete workflow from dataset preparation through model evaluation.

---

## Project Structure

    animal-vehicle-classification-cnn/
    │
    ├── main/
    │   └── <python-file>.py
    │
    ├── notebook/
    │   └── <notebook-file>.ipynb
    │
    ├── README.md
    │
    └── requirements.txt

---

## Results

The project compares two CNN architectures with different levels of complexity.

The lightweight CNN provides a faster baseline, while the deeper CNN achieves more balanced classification behavior and produces a more clearly defined prediction probability distribution.

The main experimental trade-off can be summarized as:

| Model | Architecture | Training Time | Main Characteristic |
|---|---|---:|---|
| CNN 1 | Lightweight | ~12 min | Faster training |
| CNN 2 | Deeper | ~44 min | More expressive and balanced |

The exact performance metrics produced during the experiments are available in the notebook through the generated evaluation plots, confusion matrices and classification metrics.

---

## Conclusion

This project demonstrates an end-to-end **Deep Learning and Computer Vision workflow**, covering:

- Dataset adaptation
- Binary classification formulation
- Image normalization
- Class balancing
- Data augmentation
- CNN architecture design
- Model training
- Model persistence
- Performance evaluation
- Confusion matrix analysis
- Probability analysis
- Classification threshold optimization
- Model comparison
- Error analysis

The comparison between the two architectures demonstrates that model selection involves more than simply maximizing accuracy.

Training time, model complexity, classification behavior, confidence, and application requirements must all be considered when designing a machine learning system.

The project also highlights the importance of critically evaluating the relationship between a benchmark dataset and the intended deployment environment.

While CIFAR-10 is useful for demonstrating the computer vision pipeline, a real-world road-safety system would require a much more representative dataset, higher-resolution imagery, object detection capabilities, and extensive validation before deployment.

---

## Author

**Andrea Luigi Pala**

Machine Learning / AI Portfolio Project
