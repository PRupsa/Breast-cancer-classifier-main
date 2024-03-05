# README.md

---

# Ultrasound Breast Cancer Classification

Welcome to the Ultrasound Breast Cancer Classification repository. This project is an application of Convolutional Neural Networks (CNNs) for classifying ultrasound breast images into malignant or benign categories. The model was trained using TensorFlow and Keras libraries, and it uses image augmentation and batch normalization for improving the model's performance.


## Dataset

The dataset used for this project is the [Breast Cancer Dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset) from Kaggle. It consists of ultrasound images of breast cancer labeled as malignant or benign.

## Project Files

This repository includes the following files:

1. `Breast Cancer.ipynb`: This is the main Jupyter notebook file that contains all the Python code for the project.
2. `best_model.h5`: This is the pre-trained model file that can be loaded directly to make predictions.

## How the Project Works

The project involves the following steps:

1. Loading and preprocessing of ultrasound images for training and validation.
2. Defining and compiling the CNN model architecture.
3. Training the CNN model using the training data, while also validating using the validation data. This involves several callbacks for early stopping, model checkpointing, and learning rate reduction.
4. Evaluating the model's performance using the validation data, and generating the classification report and confusion matrix.

## How to Run the Project

- Clone this repository to your local machine.
- Navigate to the directory of the project.
- You can run the `Breast Cancer.ipynb` notebook to see how the model is trained. However, if you simply want to load the pre-trained model and make predictions, you can load the `best_model.h5` file using TensorFlow and Keras.

```python
from tensorflow.keras.models import load_model
model = load_model('best_model.h5')
```
- After loading the model, you can use it to make predictions.

## Prerequisites

- This project requires Python 3. Python can be downloaded from [here](https://www.python.org/downloads/).
- The project also requires the following Python libraries: os, cv2, numpy, matplotlib, tensorflow, sklearn. These can be installed using pip.

## Contributing

- We love to have your help to make this project better. All contributions are welcome!
- For major changes, please open an issue first to discuss what you would like to change.
- Please make sure to update tests as appropriate.

## License

- This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).

---

We hope you find this project useful for your learning or research in breast cancer classification using deep learning. Your satisfaction is our priority. Happy learning!
