# Formula 1 Tire Degradation Classification

Classifying which Pirelli tire compound (Soft, Medium, or Hard) an F1 car is running from its corner telemetry alone, using a CNN + BiLSTM in PyTorch. Built for APS360 (Applied Fundamentals of Machine Learning) at the University of Toronto.

The motivation: in a sport where milliseconds decide races, strategists want to track tire behaviour and degradation. High-frequency telemetry captures complex interactions between driver, car, and conditions that are hard to model by hand, so deep learning can learn the relevant features directly from the data.

## Overview

- **Data:** Race telemetry for the 2023, 2024, and 2025 seasons from the [FastF1](https://docs.fastf1.dev/) library.
- **Task:** 3-class classification of tire compound (Hard / Medium / Soft) per corner sample.
- **Models:** A Random Forest baseline and a CNN + BiLSTM neural network (60,707 trainable parameters).
- **Evaluation:** Accuracy and macro F1 (the classes are imbalanced), plus a final test on the unseen 2025 season.

```
FastF1 API -> corner-level filtering + processing -> normalized corner telemetry
(Speed, Throttle, Brake, Gear) + static features (tire age, fuel load)
-> CNN + BiLSTM -> tire compound prediction
```

## Data processing

Each sample is one driver's pass through one corner on one lap.

1. **Load and combine.** For every race in 2023 and 2024, telemetry is loaded per driver, with lap number and cumulative track distance added, and all of a driver's laps combined into one DataFrame.
2. **Fuel load estimate.** Fuel load is not published, so it is reverse-integrated from the known race-ending fuel amount using a throttle- and RPM-dependent fuel flow function from the 2023 FIA technical regulations (with a density conversion from litres to kg).
3. **Lap filtering.** Only valid racing laps are kept: quick laps, no in/out laps, fully green-flag laps, and dry compounds only (Soft, Medium, Hard).
4. **Corner windows.** Each corner's telemetry zone is defined using its neighboring corners, with entry detected from braking or a downshift and exit from a sustained rise in throttle and gear. This limits wasted data points.
5. **Corner filtering.** Corners are dropped when a car ahead was within 20 m (dirty air disrupts the approach) or when DRS was active.
6. **Resampling and normalization.** Speed, throttle, brake, and gear are each normalized to 100 points per corner and stored alongside the static features.

Each sample contains:

| Feature | Description |
| --- | --- |
| Speed, Throttle, Brake, Gear | 100-point time series through the corner |
| TireAge | Laps on the current tire |
| FuelLoad | Estimated fuel mass (kg) |
| TireCompound | Label: Hard, Medium, or Soft |

**Dataset size:** 577,424 samples from 2023 and 2024 (roughly 55% Hard, 35% Medium, 10% Soft). The 2025 season (296,290 samples) is held out completely as new, unseen test data.

**Splits.** The 2023-24 data is split by race rather than by random row, so no race appears in more than one set. A randomized search selects race subsets so that train, validation, and test all keep the same class distribution (about 410k / 81k / 86k samples).

## Models

### Baseline: Random Forest

Chosen because it needs minimal data processing and feature encoding, handles nonlinear relationships, and reports feature importance, which is useful for sanity-checking that the problem is learnable. Input is summary statistics (mean, min, max, std) of each of the four telemetry channels, plus tire age and fuel load. Hyperparameters came from a randomized search on macro F1: 400 trees, max depth 15, min samples per leaf 3, max features 0.5, balanced class weights.

### Neural network: CNN + BiLSTM

- **CNN stage:** Conv1d (4 to 32 channels, kernel 5) with BatchNorm, ReLU, and MaxPool picks out features such as throttle spikes.
- **BiLSTM stage:** A bidirectional LSTM (hidden size 64) reads those features in both temporal directions for broader context. This two-stage design follows prior work on CNN-LSTM time series classification.
- **Static branch:** Tire age and fuel load pass through a small MLP and are concatenated with the LSTM output.
- **Head:** Two linear layers (144 to 64 to 3) with ReLU and dropout produce the class prediction.
- **Training:** AdamW with weight decay, class-weighted oversampling, a learning-rate scheduler, and early stopping on validation macro F1.

The notebook documents 11 training trials covering class-weighted loss, oversampling, model capacity, learning rate, and weight decay. The final model is **Trial 5**, picked as the smallest, most efficient model with approximately the best validation macro F1.

| Trial | Accuracy | Macro F1 | Hard F1 | Medium F1 | Soft F1 | Soft recall | Params |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 (final) | 0.65 | 0.54 | 0.76 | 0.54 | 0.33 | 0.36 | 60,707 |
| 7 | 0.66 | 0.54 | 0.76 | 0.56 | 0.31 | 0.31 | 60,707 |
| 10 | 0.65 | 0.55 | 0.76 | 0.55 | 0.33 | 0.37 | 65,731 |

## Results

| Model | Dataset | Accuracy | Macro F1 |
| --- | --- | --- | --- |
| Random Forest (baseline) | Test (2023-24 races) | 0.645 | 0.55 |
| CNN + BiLSTM (final) | Test (2023-24 races) | 0.65 | 0.54 |
| CNN + BiLSTM (final) | 2025 season (unseen) | 0.56 | 0.47 |

Per-class F1 for the final model on the 2023-24 test set was 0.76 (Hard), 0.53 (Medium), and 0.32 (Soft). On 2025 it was 0.67 (Hard), 0.50 (Medium), and 0.22 (Soft).

Macro F1 is the headline metric because Hard dominates the dataset, so a model can post high accuracy while doing poorly on Soft.

### Qualitative findings

Errors overwhelmingly occur between **adjacent** compounds (Hard and Medium, Medium and Soft) rather than between the extremes (Hard and Soft). That matches the physical picture: compounds sit on a continuous hardness spectrum, and some conditions suit two neighboring tires equally well, while Hard and Soft share no overlapping operating window. The model seems able to place a sample along the hardness spectrum but lacks the precision to pin down the exact class. The same pattern holds on the 2025 data.

### Generalization to the 2025 season

Performance dropped from 65% to 56% accuracy on 2025. Part of that gap comes from a shift in how teams used tires: the 2025 class mix was roughly 46% Hard, 42% Medium, and 12% Soft, versus about 55% Hard, 34% Medium, and 10% Soft in the training seasons. So the drop reflects a mismatch in strategic patterns, not only an inability to learn the task.

## Discussion: why performance is limited

- **Imprecise telemetry:** sensor calibration and recording practices can vary across seasons in ways the raw signal doesn't show.
- **Tire compounds change over time:** Pirelli adjusts compound composition between seasons, so identical inputs can correspond to different physical tire behaviour.
- **Single-corner telemetry:** speed, throttle, brake, and gear in one corner may not carry enough information to separate similar compounds.
- **Ambiguous ground truth:** compound choice is a team strategy decision, so some "incorrect" predictions may be valid alternatives.
- **Cars evolve:** regulations and mid-season upgrades change the relationship between car behaviour and tires.
- **Untracked driver behaviour:** drivers adjust brake bias and differential settings to offset degradation, which hides wear from the telemetry, and that data isn't public.

**What worked:** more data, matching class distributions across all splits, and giving the model enough capacity for the input style each brought clear improvements. Oversampling had mixed results and did not fix the Soft class.

## Ethical considerations

The model is trained only on dry-weather laps from 2023-24, so it should not be treated as a general racing standard. It has no awareness of racing in the rain or of how different regulation sets affect car behaviour, and repurposing it for those conditions could be dangerous.

## Repository contents

```
APS360_Final_Project.ipynb   # Full pipeline: data processing, baseline, NN, evaluation
ModelSaves/                  # Trained model weights
README.md
```

The FastF1 cache and the processed `.pkl` datasets (`output/`) are not committed because of their size. See below for how to get them.

## Running the notebook

Requires Python 3.9+.

```bash
pip install fastf1 numba imbalanced-learn xgboost sktime scikit-learn
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pandas numpy matplotlib jupyter
```

Then open `APS360_Final_Project.ipynb` and run the cells in order. Notes:

- The first run downloads race data through FastF1 and caches it in `./cache`. Processing three seasons takes a long time, so expect hours rather than minutes.
- Processed datasets are written to `./output/` and trained weights to `./ModelSaves/`.
- To skip the slow data step, download the pre-processed data from the shared [Drive folder](https://drive.google.com/drive/folders/1NyE7pC-r9Sb6fX6SGxVMEzLjvsqSADbz?usp=sharing) and place the `.pkl` files in `output/`.
- The final model is the Trial 5 checkpoint, `SecondModel_wd1e-4_v2_bestValF1`.

## Limitations and future work

- Fuel load is an estimate derived from throttle and RPM, not a measured value.
- Corner detection is heuristic and may mis-segment unusual corners.
- Compound classification is only a proxy for degradation. Predicting a continuous measure such as lap-time loss would be a natural next step.
- Adding track temperature, circuit identity, or telemetry from multiple corners or full laps could help separate adjacent compounds.
- Training on more seasons could reduce sensitivity to year-specific car and tire changes.

## References

- Arce-Saenz, Izquierdo-Reyes, and Bustamante-Bello. Exploring single-head and multi-head CNN and LSTM-based models for road surface classification using on-board vehicle multi-IMU data. *Scientific Reports*, 15, 2025.
- Karim, Majumdar, Darabi, and Chen. LSTM fully convolutional networks for time series classification. *IEEE Access*, 6, 2018.
- Moosavi et al. Driving style representation in convolutional recurrent neural network model of driver identification. *arXiv:2102.05843*, 2021.
- Noble and Straw. Revealed: The data on F1's worsening dirty air problem. *The Race*, 2025.
- Oehrly. FastF1: A Python package for accessing and analyzing Formula 1 results, schedules, timing data and telemetry. 2025.
- Pirelli. Formula 1 and Pirelli. 2025.
- Sasikumar, Leema, and Balakrishnan. Data-driven pit stop decision support for Formula 1 using deep learning models. *Frontiers in Artificial Intelligence*, 8, 2025.
- Todd et al. Explainable time series prediction of tyre energy in Formula One race strategy. *Proceedings of the 40th ACM/SIGAPP Symposium on Applied Computing*, 2025.
- FIA. 2023 Formula 1 Technical Regulations, Issue 6. 2023.
