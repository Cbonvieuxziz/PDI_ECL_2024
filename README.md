# PDI Ecole Centrale de Lille 2024

This repository encompasses the work conducted by Antonin MOREL as part of the PDI project at Ecole Centrale de Lille, in collaboration with Gridvise.

The objective of this project is to implement various models for predicting electricity price trends.

## Project Structure

* __ANN__: The 'ANN-price-forecasting' directory contains all the source code for model development, training, and evaluation of the Artificial Neural Network model.

* __WNN__: The 'WNN-price-forecasting' directory contains all the source code for model development, training, and evaluation of the Weighted Nearest Neighbors model.

* __Event__: The 'event-price-forecasting' directory contains all the source code for the developpement of the solution to get from the news a coefficient to pass as an argument to the ANN model. It was thought to measure the impact of the environnement on the electricity price.

* __Data__ : the 'data' directory contains all the data used to train and test the models

* __Website__ : the 'web-app' directory containes all the source code of the website. Displays the result of the ANN with the energymarketprice data.

## How to Use

To test the models, install the dependencies running 

```
$ python -m pip install -r requirements.txt
```

and simply run the notebook in the corresponding directory.

For the website, you must have node installed. Go into the 'web-app' directtory and run

```
$ npm install
```

to set up the environnement and

```
$ npm run start
```

to start the developpment server