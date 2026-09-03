## Overview

This project explores the development of a **Computer Vision system for automatic image classification**, designed to distinguish between **animals and vehicles** in an urban road environment.

The project was developed for **VisionTech Solutions**, with the goal of investigating how deep learning can support automated wildlife monitoring and road-safety systems. In a potential real-world scenario, cameras installed along urban roads could use image classification to identify animals entering the roadway and trigger appropriate safety responses.

The solution is based on **Convolutional Neural Networks (CNNs)** implemented with **TensorFlow and Keras**, and trained on the **CIFAR-10 dataset**.

Rather than performing the original 10-class CIFAR-10 classification task, the dataset was adapted to the specific business problem by grouping the relevant classes into two categories:

- **Animal:** bird, cat, deer, dog, frog, horse
- **Vehicle:** automobile, truck

Airplane and ship images were excluded because they are not relevant to the terrestrial road-safety scenario.

---

## Approach

The project follows a complete deep learning workflow:

**CIFAR-10 → Dataset Filtering → Binary Labeling → Class Balancing → Normalization → Data Augmentation → CNN Training → Evaluation → Error Analysis**

The original dataset presents a significant class imbalance after grouping the original classes:

- **30,000 animal images**
- **10,000 vehicle images**

To prevent the model from becoming biased toward the majority class, the animal class was downsampled to obtain a balanced training dataset containing:

- **10,000 animal images**
- **10,000 vehicle images**

The test set was kept separate and was not downsampled, preserving its original distribution for evaluation.

---

## Implementation

The implementation was developed using **Python, TensorFlow/Keras, NumPy and Matplotlib**, with Scikit-learn used for selected evaluation metrics.

### Dataset Preparation

The CIFAR-10 dataset is loaded directly through the TensorFlow/Keras API.

The original labels are then filtered to retain only the classes relevant to the target application.

The selected classes are mapped to a binary target:

- `0` → Animal
- `1` → Vehicle

The image pixel values are normalized from the original `[0, 255]` range to `[0, 1]`.

The resulting images maintain the original CIFAR-10 shape:

`32 × 32 × 3`

representing height, width and RGB channels.

### Class Balancing

The training dataset is initially imbalanced because six CIFAR-10 classes are grouped into the animal category, while only two classes represent vehicles.

A random downsampling procedure is therefore applied to the animal samples.

A fixed random seed (`42`) is used to make the sampling reproducible.

This produces an equal number of training examples for both target classes.

### Data Augmentation

Because balancing the dataset requires discarding part of the available animal samples, **data augmentation** is used to increase the variability of the training data.

The augmentation pipeline includes:

- Random rotations
- Horizontal flips
- Width shifts
- Height shifts
- Zoom

The transformations are applied dynamically during training using Keras `ImageDataGenerator`.

This allows the network to see different variations of the training images without explicitly creating and storing additional datasets.

---

## CNN Models

Two CNN architectures were implemented to investigate the relationship between **network complexity, training time and classification performance**.

### CNN 1 — Lightweight Baseline

The first model was intentionally designed as a lightweight baseline.

Its architecture consists of:

- Input layer
- One convolutional layer with 96 filters
- Max pooling
- Flatten layer
- Single sigmoid output neuron

The model uses:

- **ReLU** activation in the convolutional layer
- **Sigmoid** activation for binary classification
- **Adam** optimizer
- **Binary Cross-Entropy** loss

This architecture was designed to provide a relatively fast training baseline and to evaluate how well a simple CNN can solve the classification problem.

### CNN 2 — Deeper Architecture

The second model increases the representational capacity of the network through multiple convolutional blocks.

The architecture includes:

- Multiple convolutional layers
- Increasing filter sizes: `32 → 64 → 128`
- Max pooling
- Batch Normalization
- Dropout
- Fully connected layer
- Sigmoid output layer

The deeper architecture is designed to learn hierarchical visual representations.

Earlier convolutional layers can capture lower-level patterns such as:

- Edges
- Colors
- Textures

Deeper layers can progressively learn more complex patterns associated with the visual structure of animals and vehicles.

**Batch Normalization** is used to stabilize the training process, while **Dropout** is introduced as a regularization technique to reduce overfitting.

---

## Training

Both models are trained using the balanced training dataset and the data augmentation pipeline.

The main training configuration includes:

- Optimizer: **Adam**
- Loss: **Binary Cross-Entropy**
- Metric: **Accuracy**
- Batch size: **64**
- Epochs: **10**

The test dataset is provided as validation data during training to monitor the model's performance on unseen examples.

Training histories are stored in JSON format, allowing the evolution of accuracy and loss to be analyzed after training.

Model weights are also saved in `.weights.h5` format, making it possible to reload trained models without repeating the complete training process.

---

## Model Persistence

The implementation includes functionality for both **saving and loading model artifacts**.

Training histories are saved as JSON files, while neural network weights are saved using Keras' `.weights.h5` format.

The notebook also contains optional functionality for downloading previously trained weights from Google Drive using `gdown`.

This makes it possible to separate the computationally expensive training phase from subsequent inference and evaluation.

---

## Evaluation

The models are evaluated using multiple perspectives rather than relying exclusively on accuracy.

The evaluation includes:

- Accuracy
- Precision
- Recall
- Binary Cross-Entropy Loss
- Confusion Matrix
- Prediction probability distributions
- Training and validation curves
- Classification threshold analysis

### Confusion Matrix

The confusion matrix is used to identify the types of classification errors made by the network.

This is particularly important for the target application because the cost of different errors may not be equivalent.

For example, incorrectly classifying an animal as a vehicle and incorrectly classifying a vehicle as an animal could have different operational consequences.

### Prediction Probabilities

The model's sigmoid output is interpreted as the predicted probability of the `vehicle` class.

The distribution of these probabilities is visualized to investigate how confidently the network separates the two categories.

---

## Classification Threshold Analysis

The default classification threshold is `0.5`.

However, the implementation also evaluates alternative thresholds:

- `0.3`
- `0.5`
- `0.7`
- `0.9`

For each threshold, **precision and recall** are calculated.

This experiment demonstrates that the classification threshold is itself an important decision parameter.

Changing the threshold modifies the balance between false positives and false negatives.

In a real-world road-safety application, this threshold could potentially be optimized according to the business objective and the relative cost of different types of errors.

---

## Model Comparison

The two architectures highlight a clear engineering trade-off.

### CNN 1

The lightweight model:

- Has a simpler architecture
- Requires less computational effort
- Trains significantly faster
- Provides a useful baseline for comparison

Its behavior is more aggressive toward the vehicle class, resulting in a different balance of classification errors.

### CNN 2

The deeper model:

- Uses substantially more convolutional layers
- Learns richer visual representations
- Uses Batch Normalization and Dropout
- Requires significantly more training time
- Produces more balanced classification behavior

The second architecture therefore provides a useful example of the trade-off between **computational cost and model capacity**.

---

## Key Insights

The experiments highlight several important observations.

### 1. Data Preparation Has a Major Impact

The original CIFAR-10 labels were not directly suitable for the target business problem.

Adapting the dataset to the real-world scenario required:

- Removing irrelevant classes
- Grouping multiple classes
- Converting the problem into binary classification
- Addressing class imbalance

This preprocessing step is an important part of the machine learning pipeline rather than simply a preliminary operation.

### 2. Class Imbalance Must Be Considered

Without balancing, a model could achieve apparently strong accuracy while being biased toward the majority animal class.

Downsampling combined with data augmentation provides a simple strategy to mitigate this issue.

### 3. Model Complexity Comes With a Cost

Increasing the depth of the CNN improves its ability to learn complex visual representations, but significantly increases training time.

This highlights an important engineering consideration: the most complex model is not necessarily the most appropriate model for every deployment scenario.

### 4. Accuracy Is Not Enough

The confusion matrix, precision, recall and probability distributions provide a more informative view of model behavior.

For an application involving road safety, understanding the type of errors made by the model can be more important than considering overall accuracy alone.

### 5. The Classification Threshold Is a Business Parameter

The threshold used to convert the predicted probability into a binary class can be adjusted according to the desired operational behavior.

This creates a direct connection between model outputs and application requirements.

---

## Limitations

Despite the promising results, the system should be considered a **proof of concept** rather than a production-ready autonomous driving component.

The main limitation is the dataset itself.

CIFAR-10 images have a resolution of only **32 × 32 pixels** and do not reproduce the complexity of real-world road environments.

A real deployment would need to handle factors such as:

- Higher image resolution
- Different lighting conditions
- Night-time environments
- Weather conditions
- Motion blur
- Occlusion
- Different camera angles
- Object scale and distance
- Multiple objects in the same frame
- Complex urban backgrounds

Furthermore, the current solution performs **image-level classification** rather than object detection.

It determines whether an image belongs to the animal or vehicle category, but it does not determine the exact location of an object within an image.

---

## Future Development

A production-oriented version of this system could be extended through:

- Higher-resolution and domain-specific datasets
- Transfer learning with pretrained architectures
- More advanced CNN architectures
- Object detection models
- Real-time inference
- Model quantization and optimization
- Confidence calibration
- Automated threshold optimization
- Evaluation under different environmental conditions
- Integration with real-time camera streams

A particularly relevant next step would be replacing the CIFAR-10 dataset with a **domain-specific road-scene dataset**, allowing the model to learn from images that better represent the actual deployment environment.

---

## Technologies

- **Python**
- **TensorFlow**
- **Keras**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**
- **Google Colab**
- **gdown**

---

## Project Structure

The repository contains the complete implementation and the original experimental notebook.

The main components are:

- `main/` — Python implementation
- `notebook/` — Google Colab/Jupyter notebook containing the complete experimentation workflow
- `README.md` — Project documentation
- `requirements.txt` — Python dependencies

---

## Conclusion

This project demonstrates an end-to-end **Deep Learning and Computer Vision workflow**, covering dataset adaptation, preprocessing, class balancing, data augmentation, CNN architecture design, training, model persistence, evaluation, threshold optimization and error analysis.

The comparison between a lightweight CNN and a deeper architecture also demonstrates how model design involves more than simply maximizing predictive performance: **training time, computational complexity, class-specific errors and application requirements must all be considered when selecting a model**.

The project therefore provides a practical example of applying deep learning to a realistic business problem while critically evaluating both the capabilities and limitations of the resulting system.
