# COVID-19-Prediction-in-India

## About:

The code predicts the number of COVID-19 cases and deaths in India for the particular date which is provided. Implemented using TensorFlow.

## Developed by:

[Vigneshwar Ravichandar](https://github.com/ToastCoder)

## Execution Instructions:

### Windows without WSL:

1. Execute the train_cases.py and train_deaths.py for retraining the model.

2. Execute the visualize.py for visualizing the graph and accuracy.

3. Execute the test.py for predicting cases and deaths for specific date.

### Windows(WSL) / Linux / MacOS:

There is a script which automates the entire workflow called autorun.sh.

1. On a WSL Bash/Linux Terminal/Mac Terminal, type **bash autorun.sh**.

2. Press Y for executing the script or Press N for cancelling the execution.

![img1](https://github.com/ToastCoder/COVID-19-Prediction-in-India/blob/master/images/image1.png)

3. The script will check for dependencies.

4. Press Y for retraining the model or Press N for skipping it.

![img2](https://github.com/ToastCoder/COVID-19-Prediction-in-India/blob/master/images/image2.png)

5. Press Y for visualizing the model's graph and accuracy or Press N for skipping it.

![img3](https://github.com/ToastCoder/COVID-19-Prediction-in-India/blob/master/images/image3.png)

6. Now, enter the date of your choice to predict.

![img4](https://github.com/ToastCoder/COVID-19-Prediction-in-India/blob/master/images/image4.png)

7. The code will predict the number of expected cases and deaths for the specified date.

![img5](https://github.com/ToastCoder/COVID-19-Prediction-in-India/blob/master/images/image5.png)
