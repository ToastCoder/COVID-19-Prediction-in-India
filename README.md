# COVID 19 Prediction in India

### About:  
 The code predicts the number of COVID-19 cases and deaths in India for the respective date which is provided. It is implemented using TensorFlow. There are 2 models where is one model is used to predict the number of cases and the other one is used to predict the number of deaths. Both of these models are achieving a maximum accuracy of 99.95%. 

### Tested with: 
#### Operating System:
  * Pop OS 20.10 
#### Python version:
  * Python 3.8.6 64-bit 
#### Packages:
  * tensorflow 2.4.1
  * numpy 1.19.5
  * pandas 1.2.4
  * matplotlib 3.4.2
  * scipy 1.6.3
  * sklearn 0.24.2
  * datetime 

### Developed by:  
 [Vigneshwar Ravichandar](https://github.com/ToastCoder)

### Explanation: 
  Regression is the concept used here. Regression is a statistical method used in finance, investing, and other disciplines that attempts to determine the strength and character of the relationship between one dependent variable (usually denoted by Y) and a series of other variables (known as independent variables). Regression is used here where the relation between No of days (X) and Cases/Deaths (Y) is found by the model and the unknown Cases/Deaths (Y) is found for any value of No of Days (X). 

### About the Neural Network: 
  A typical form of Artificial Neural Network (ANN) is used here. The layers of the Neural Network Architecture is as follows... 
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 2)                 4         
_________________________________________________________________
dense_1 (Dense)              (None, 79)                237       
_________________________________________________________________
dense_2 (Dense)              (None, 79)                6320      
_________________________________________________________________
dense_3 (Dense)              (None, 79)                6320      
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 80        
=================================================================
Total params: 12,961
Trainable params: 12,961
Non-trainable params: 0
``` 
The ANN is fitted with the normalized data done using `StandardScaler()`. The model is trained with the `huber` loss function and `adamax` optimizer. 

### Execution Instructions:  
 Execute the following command in the terminal to run with default procedure.

```python
python3 main.py --test=True
```

### Command Line Arguments:

* `-tr` (or) `--train` - Used to train the Neural Network.  
  * **Argument type:** str  
  * **Parameter type:** Optional  
  * **Values:**  
    * `cases` - Used for training the model_cases only.
    * `deaths` - Used for training the model_deaths only.
    * `all` - Used for training both the models.
  * **Default value:** "none"

* `-t` (or) `--test` - Used to test the Neural Network with custom inputs.
  * **Argument type:** bool  
  * **Parameter type:** Mandatory 
  
* `-v` (or) `--visualize` - Used to vizualize the metrics.
  * **Argument type:** bool  
  * **Parameter type:** Optional
  * **Default value:** False
  
* `-req` (or) `--install_requirements` - Used to install the required dependancies.
  * **Argument type:** bool  
  * **Parameter type:** Optional
  * **Default value:** False

* `-e` (or) `--epochs` - Used for mentioning the number of epochs for both of the models.
  * **Argument type:** int
  * **Parameter type:** Optional
  * **Default value:** 500

* `-bs` (or) `--batch_size` - Used for mentioning the batch size for both of the models.
  * **Argument type:** int
  * **Parameter type:** Optional
  * **Default value:** 150

* `-l` (or) `--loss` - Used for mentioning the loss function for both of the models.
  * **Argument type:** str
  * **Parameter type:** Optional
  * **Default value:** "huber"

* `-op` (or) `--optimizer` - Used for mentioning the optimizer for both of the models.
  * **Argument type:** str
  * **Parameter type:** Optional
  * **Default value:** "adamax"

### Images:

![img1](https://github.com/ToastCoder/COVID-19-Prediction-in-India/blob/master/images/img1.png)

                              *Screenshot mentioning the training command* 

![img2](https://github.com/ToastCoder/COVID-19-Prediction-in-India/blob/master/images/img2.png)

                              *Screenshot mentioning the testing command*  

![img3](https://github.com/ToastCoder/COVID-19-Prediction-in-India/blob/master/images/img3.png)

                            *Screenshot mentioning the visualizing command*  

