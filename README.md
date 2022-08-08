# Predicting Soil Reaction Front Using Hybrid Neural Network

This repo contains source codes, input data, and an example output for the hybrid neural network proposed in the article Wen et al. (2022), which targets improving the accuracy of a physics-based model in simulating soil reaction front using the neural network.

#### Reference
Wen, T., Chen, C., Zheng, G., Bandstra, J., and Brantley, S.L. (2022). Using a Neural Network – Physics-based Hybrid Model to Predict Soil Reaction Fronts. _Computers & Geosciences_, https://doi.org/10.1016/j.cageo.2022.105200



## Installation

0. Download this repo as a zip file to your local computer. Unzip the downloaded file.
1. Make sure all of the dependencies are installed before running the HNN codes. The list of dependencies is listed below

#### Dependencies

| __Required Package__ | __Version__ |
|----------------------|-------------|
| Python               | __3.8.3__   |
| sympy                | 1.10.1      |
| tensorflow           | 2.8.0       |
| pandas               | 1.3.5       |
| numpy                | 1.22.3      |



## Get Started

0. Make sure all of the above dependencies are installed before running the code

1. Specify parameters in the final.py
Hyperparameters:
```
WHETHER_TRAIN = True  ## Training the model from scratch. If using the pre-trained model, please set it to False.
ONLYBEST = True       ## Only train/test the best performance models ['7a', '7b', '9a', '11a', '11c', '14a','14b']; Set to False if training all models.
HOME_DIR = "./test"   ## Define the directory where the result will be saved.
```

2. Input data: Input data is saved in data/all_data.csv

3. Run the finaly.py script.

```
python final.py
```


## License
This work is licensed under the [MIT license](https://github.com/HANDS-Research-Group/HNN_Soil_Reaction_Front/blob/main/LICENSE).
