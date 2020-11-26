#!/bin/sh
# Author: Vigneshwar Ravichandar
# Autorun shell script to ensure requirements,train the model and test the model.

echo "Autorun.sh verifies the requirements and runs the code properly"
echo "To proceed enter Y, To cancel press N"
read response
if [[ $response == "y" || $response == "Y" ]]
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
    pip3 install pickle
    pip3 install matplotlib
    pip3 install datetime
    python3 train.py
    python3 test.py
fi
echo "Thank you for watching!"