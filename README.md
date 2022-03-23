# HNN_Soil_Reaction_Front

This is the implementation of hybrid neural network for soil reaction front prediction.

Please use the code as:

```
python final.py
```

Hyperparameters:
```
WHETHER_TRAIN = True  ## Training the model from scratch. If using pre-trained model, please set to False.
ONLYBEST = True       ## Only train/test the best performance models ['7a', '7b', '9a', '11a', '11c', '14a','14b']; Set to False if training all models.
HOME_DIR = "./test"   ## Define the directory where result will be saved.
```

Data is saved in data/all_data.csv