# Platform Supplier Rationales

This repository contains the code and data for my master thesis on "Naive vs Smart Suppliers in Two-sided Online Marketplaces For Different Price-Setting Entities".

From the `pipeline.py` file, all important functions can be called. Before, the required folder structure must be set up.

## Folder Structure

- `drl_data`: Contains modules for data generation.
  - `data_generation.py`: Functions for generating training, validation, and testing data.
  - `testing_data.py`: Empty folder for the testing data.
  - `testing_data.py`: Empty folder for the stable testing data.
  - `training_data.py`: Empty folder for the training data.
  - `validation_data.py`: Empty folder for the vaidation data.
- `sb3`: Contains modules for training and testing the DRL algorithms.
  - `algos`: Different DRL training implementations.
  - `environments`: Environments for different algorithms.
  - `testing`: Different algorithm testing implementations. This is equal to scenario 1.
  - `logs`: Empty folder for training logs.
- `simulations`: Contains simulation modules.
  - `classes.py`: Basic classes for suppliers, buyers and matches.
  - `simulation.py`: Functions for running simulations.
- `results`: Contains modules for hypothesis testing.
  - `hypothesis_testing.py`: Functions for hypothesis test creation.
  - `figure_plotting.ipynb`: Functions for creating plots.
  - `plots`: Empty folder structure for plots.
    - `simulation`
      - `difference`
      - `absolute`
    - `testing`
      - `difference`
      - `absolute`

## Important Functions in `pipeline.py`

1. **Data Generation**:
   - `generate_data`: Generates data for training, validation, and testing.
   - `create_stable_testing_data`: Creates stable testing data.

2. **Training**:
   - `train_centralized`: Trains the centralized algorithm.
   - `train_hybrid`: Trains the hybrid algorithm.
   - `train_decentralized`: Trains the decentralized algorithm.

3. **Testing**:
   - `test_centralized`: Tests the centralized algorithm.
   - `test_hybrid`: Tests the hybrid algorithm.
   - `test_decentralized`: Tests the decentralized algorithm.

4. **Simulation**:
   - `simulate`: Runs simulations with different setups.

5. **Hypothesis Testing**:
   - `create_hypothesis_test_testing`: Creates hypothesis tests for testing data.
   - `create_hypothesis_test_simulation`: Creates hypothesis tests for simulation data.

## Usage

To run the pipeline, execute the `pipeline.py` script. This will perform the following steps:
1. Data Generation
2. Algorithm Training
3. Algorithm Testing
4. Simulation
5. Hypothesis Testing
