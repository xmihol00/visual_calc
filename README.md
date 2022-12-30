# Visual Calculator

## Training, Validating and Testing Data
upload and download the data using this link, do not puth them on GitHub:
https://drive.google.com/drive/folders/1yFQTQFaheAh_cH8AzxKYhjMwJ06dxA9t?usp=sharing

### Usage
* Download the dataset from https://www.kaggle.com/datasets/michelheusser/handwritten-digits-and-operators and extract it in the data/ folder
* Run training.py to train and save the model
* Run recognition.py to test the trained model on a static image of a simple equation
* Run live_recognition.py to run the trained model on a live camera feed (The camera index needs to bet set as cv2.VideoCapture(x). This is usually 0 but can be different depending on setup)