import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import csv

# data_path = 'C:/Users/roei.w/Desktop/machine/Input/validation/'
with open('C:/Users/roei.w/Desktop/machine/Input/tmp11.csv', 'r') as csvin, open('C:/Users/roei.w/Desktop/machine/Input/tmp.csv', 'w') as csvout:
    reader = csv.reader(csvin)
    writer = csv.writer(csvout, lineterminator = '\n')
    i=0
    for row in reader:
        print(row)
        print(len(row))
        print(row[9])



