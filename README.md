# HNN_Soil_Reaction_Front

This repo contains source codes, input data, and example output for the hybrid neural network proposed in the article Wen et al. (XXXX), which targets at improving the accuracy of physics-based model in simulating soil reaction front using the neural network.

#### Reference
Wen, T., Chen, C., Zheng, G., Bandstra, J., and Brantley, S.L. (XXXX). Using a Neural Network – Physics-based Hybrid Model to Predict Soil Reaction Fronts. _Under Revision_, https://doi.org/XXXXXXX



## Installation

0. Download this repo as a zip file to your local computer. Unzip the downloaded file.
1. Make sure all of the dependencies are installaed before running the HNN codes. The list of dependencies is listed below

### Dependencies

| __Required Package__ | __Version__ |
|----------------------|-------------|
| Python               | __3.8.3__   |
| sympy                | 1.10.1      |
| tensorflow           | 2.8.0       |
| pandas               | 1.3.5       |
| numpy                | 1.22.3      |



## Get Started

0. Make sure all of above dependencies are installed before running the code

1. Specify parameters in the final.py
Hyperparameters:
```
WHETHER_TRAIN = True  ## Training the model from scratch. If using pre-trained model, please set to False.
ONLYBEST = True       ## Only train/test the best performance models ['7a', '7b', '9a', '11a', '11c', '14a','14b']; Set to False if training all models.
HOME_DIR = "./test"   ## Define the directory where result will be saved.
```

2. Input data: Input data is saved in data/all_data.csv

3. Run the finaly.py script.

```
python final.py
```


## License
This work is license under the [MIT license](https://github.com/HANDS-Research-Group/HNN_Soil_Reaction_Front/blob/main/LICENSE).
