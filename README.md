#  Deep Neural Network (DNN) for MNIST Digit Classification

This project implements a **Deep Neural Network (DNN)** using **TensorFlow** to classify handwritten digits from the **MNIST dataset**.  
It demonstrates data preprocessing, model design, training, evaluation, and performance improvement techniques.

---

##  Project Overview

- **Dataset**: [MNIST Handwritten Digits](http://yann.lecun.com/exdb/mnist/)
- **Objective**: Classify digits (0–9) using a neural network.
- **Approach**: Build a DNN from scratch with TensorFlow/Keras and evaluate performance.

---

##  Plan of Action

1. **Read the Data**
   - Load the MNIST dataset from `tensorflow.keras.datasets`.

2. **Preprocessing**
   - Normalize image pixel values by dividing by the maximum (255).
   - Flatten the images for feeding into a fully connected network.

3. **Neural Architecture Design**
   - **Input Layer**: Flattened 784 neurons  
   - **Hidden Layer**: 128 neurons, ReLU activation  
   - **Output Layer**: 10 neurons, Softmax activation  

4. **Compile**
   - **Optimizer**: Adam  
   - **Loss Function**: Categorical Cross-Entropy  
   - **Metric**: Accuracy  

5. **Training**
   - 20% validation split  
   - 5 epochs  

6. **Testing**
   - Evaluate model performance on unseen test data.

7. **Visualizations**
   - Plot training/validation accuracy and loss curves.  
   - Display sample predictions and a **confusion matrix**.

8. **Extensions**
   - Deeper neural network (4–5 layers).  
   - Add **Dropout** layers for regularization.  
   - Add **L2 regularization** for better generalization.

---

##  Model Architecture Summary

| Layer (Type)     | Output Shape | Parameters | Activation |
|------------------|---------------|-------------|-------------|
| Flatten          | (None, 784)   | 0           | -           |
| Dense            | (None, 128)   | 100,480     | ReLU        |
| Dense (Output)   | (None, 10)    | 1,290       | Softmax     |

---

