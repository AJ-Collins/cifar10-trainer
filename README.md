# CNN Image Classification with CIFAR-10

This project implements a **Convolutional Neural Network (CNN)** to classify images from the **CIFAR-10** dataset. The model is trained using TensorFlow/Keras and evaluates its performance using accuracy, loss plots, and a confusion matrix.

## Dataset
The **CIFAR-10 dataset** consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. It is divided into **50,000 training images** and **10,000 test images**.

### Class Labels:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Installation & Setup
### Prerequisites
Make sure you have Python installed. It's recommended to use a virtual environment.

1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # For Linux/macOS
   venv\Scripts\activate  # For Windows
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Training the Model
Run the following command to train the CNN model on CIFAR-10:
```sh
python cnn_train.py
```

This will load the dataset, preprocess it, define the CNN architecture, train the model, and evaluate its performance.

## Model Architecture
The CNN consists of:
- **Convolutional Layers** for feature extraction
- **Batch Normalization** to stabilize training
- **MaxPooling Layers** for downsampling
- **Dropout Layers** to prevent overfitting
- **Dense Layers** for classification

## Evaluation Metrics
After training, the model is evaluated on the test set using:
- **Accuracy & Loss Plots** (Matplotlib)
- **Confusion Matrix** (Seaborn & Scikit-learn)
- **Classification Report** (Precision, Recall, F1-score)

## Results
The model achieves a test accuracy of approximately **77%**.

## Future Improvements
- Experiment with deeper architectures.
- Use data augmentation to improve generalization.
- Implement Transfer Learning with pre-trained models like ResNet or VGG.

## Contributing
Feel free to fork this repo and submit pull requests to improve it.
