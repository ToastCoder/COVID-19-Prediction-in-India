#!/bin/sh
# Author: Vigneshwar Ravichandar
# Autorun shell script to ensure requirements,train the model and test the model.

echo "Autorun.sh verifies the requirements and runs the code properly"
echo "To proceed enter Y, To cancel press N"
read response
if [[ $response == "y" ]] || [[ $response == "Y" ]]
then

    if which python3 >/dev/null;
    then
        sudo apt install python3 -y
    fi

    if which pip3 >/dev/null;
    then
        sudo apt install python3-pip -y 
    fi

    pip3 install numpy
    pip3 install pandas
    pip3 install sklearn
    pip3 install tensorflow
    pip3 install matplotlib
    pip3 install datetime

    echo "Do you want to retrain the model?(Y/N)"
    read resp1
    if [[ $resp1 == 'y' ]] || [[ $resp1 == 'Y' ]]
    then
        echo "Retraining model..." 
        python3 train_cases.py
        python3 train_deaths.py
    elif [[ $resp1 == 'n' ]] || [[ $resp1 == 'N' ]]
    then
        echo "Skipped retraining model"
    fi

    echo "Do you want to visualize metrics?(Y/N)"
    read resp2
    if [[ $resp2 == 'y' ]] || [[ $resp2 == 'Y' ]]
    then
        python3 visualize.py
    elif [[ $resp2 == 'n' ]] || [[ $resp2 == 'N' ]]
    then
        echo "Skipped visualization"
    fi

    echo "Running test file..."
    python3 test.py

elif [[ $response == "n" ]] || [[ $response == "N" ]]
then
    echo "File execution cancelled..."
fi

echo "Thank you for executing!"
