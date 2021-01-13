# COVID 19 Prediction in India

## About:  
 The code predicts the number of COVID-19 cases and deaths in India for the respective date which is provided. It is implemented using TensorFlow. There are 2 models where is one model is used to predict the number of cases and the other one is used to predict the number of deaths. Both of these models are achieving a maximum accuracy of 99.8%. 

## Supported Operating Systems:  
 Runs on Windows, Linux and MacOS.

## Tested with:  
* Python 3.8.5 64-bit
* TensorFlow 2.4.0
* Pop OS 20.04 LTS

## Developed by:  
 [Vigneshwar Ravichandar](https://github.com/ToastCoder)

## Execution Instructions:  
 Execute the following command in the terminal to run with default procedure.

```python
python3 main.py --test=True
```

## Command Line Arguments:

* `-tr` (or) `--train` - Used to train the Neural Network.  
  * **Argument type:** str  
  * **Parameter type:** Optional  
  * **Values:**  
    * `cases` - Used for training the model_cases only.
    * `deaths` - Used for training the model_deaths only.
    * `all` - Used for training both the models.

* `-t` (or) `--test` - Used to test the Neural Network with custom inputs.
  * **Argument type:** bool  
  * **Parameter type:** Mandatory 
  
* `-v` (or) `--visualize` - Used to vizualize the metrics.
  * **Argument type:** bool  
  * **Parameter type:** Optional
  
* `-req` (or) `--install_requirements` - Used to install the required dependancies.
  * **Argument type:** bool  
  * **Parameter type:** Optional


## Images:

![img1](https://github.com/ToastCoder/COVID-19-Prediction-in-India/blob/master/images/img1.png)

  *Screenshot mentioning the training command* 
