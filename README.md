# HNN_Soil_Reaction_Front

This is the implementation of hybrid neural network for soil reaction front prediction.

## Dependencies

| __Required Package__ | __Version__ |
|----------------------|-------------|
| Python               | __3.8.3__   |
| sympy                | 1.10.1      |
| tensorflow           | 2.8.0       |
| pandas               | 1.3.5       |
| numpy                | 1.22.3      |



## How to run the code
0. Make sure all of above dependencies are installed before running the code

1. Specify parameters in the final.py
Hyperparameters:
```
WHETHER_TRAIN = True  ## Training the model from scratch. If using pre-trained model, please set to False.
ONLYBEST = True       ## Only train/test the best performance models ['7a', '7b', '9a', '11a', '11c', '14a','14b']; Set to False if training all models.
HOME_DIR = "./test"   ## Define the directory where result will be saved.
```

2. Run the finaly.py script.

```
python final.py
```

## Input Data
Input data is saved in data/all_data.csv
