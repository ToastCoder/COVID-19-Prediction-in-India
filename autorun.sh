#!/bin/sh
# Author: Vigneshwar Ravichandar
# Autorun shell script to ensure requirements,train the model and test the model.

sudo apt install python3 -y
sudo apt install python3-pip -y 
pip3 install numpy
pip3 install pandas
pip3 install sklearn
pip3 install pickle
pip3 install matplotlib
pip3 install datetime
python3 train.py
python3 test.py