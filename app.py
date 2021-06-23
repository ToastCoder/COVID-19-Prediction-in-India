# IMPORTING REQUIRED PACKAGES
import requests
from flask import Flask, render_template, redirect, url_for, request,jsonify
from werkzeug.wrappers import Request, Response

import json

# for data loading and transformation
import numpy as np 
import pandas as pd

# for statistics output
from scipy import stats
from scipy.stats import randint