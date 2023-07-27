# Digit Recognizer Using CNN in Python

This repository contains Python code for a Convolutional Neural Network (CNN) model that recognizes handwritten digits. The model is trained on the MNIST dataset and can accurately classify digits from 0 to 9.

## Overview

This digit recognizer leverages the power of deep learning and the Keras library to create and train a CNN model. The model is designed with Conv2D layers for feature extraction, MaxPooling2D layers for down-sampling, and fully connected Dense layers for classification. The code also includes steps for loading the test dataset, generating predictions, and creating a submission file.

...

## Model Training and Evaluation

The Convolutional Neural Network (CNN) model was trained on the training dataset and evaluated on the validation dataset. After training for 10 epochs, the model achieved an impressive accuracy of 98% on the validation set. This high accuracy demonstrates the model's ability to accurately recognize handwritten digits and showcases the success of our approach.

...

## Requirements

Make sure you have the following libraries installed:

- pandas
- numpy
- matplotlib
- scikit-learn
- keras

You can install the required libraries using `pip`:

```bash
pip install pandas numpy matplotlib scikit-learn keras
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your-username/Digit-Recognizer-Using-CNN.git
cd Digit-Recognizer-Using-CNN
```

2. Prepare the data:

Ensure that you have the `train.csv` and `test.csv` files in the same directory as the script. The `train.csv` file contains the labeled training data, while `test.csv` contains the unlabeled test data.

3. Train the CNN model:

Run the `training-model.py` script to train the CNN model on the training dataset. The model will be saved as `trained_model.h5` after training is complete.

```bash
python training-model.py
```

4. Evaluate the model:

The script will automatically evaluate the trained model on the validation set and display the validation loss and accuracy.

5. Generate predictions:

Run the `generate_predictions.py` script to generate predictions for the test dataset using the trained model. The predictions will be saved in a `submission.csv` file.

```bash
python generate_predictions.py
```

## Results

The `submission.csv` file will contain the predicted labels for the test dataset in the required format for submission to the competition.

## Contributing

Contributions to this repository are welcome. If you have suggestions for improvements or bug fixes, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
