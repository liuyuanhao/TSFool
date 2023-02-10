# TSFool: Crafting Highly-imperceptible Adversarial Time Series through Multi-objective Black-box Attack to Fool RNN Classifiers

## Quick Start

The Python module of TSFool, the pre-trained RNN classifiers
and the crafted adversarial sets are available

<!-- ```python-->
<!-- adv_X, adv_Y, target_X = TSFool(model, X, Y, K=2, T=30, F=0.1, eps=0.01, N=20, P=0.9, C=1, target=-1, details=False)-->
<!-- ```-->


## Update

#### February 2023:
- A new version of TSFool implemented as a Python module is available now.
- The corresponding information would be updated soon (expected to complete before Feb. 9, AOE time).


#### September 2022:
- ~~The raw experiment records about **1) the datasets selected from the UCR archive**, **2) the target LSTM classifiers**, **3) the intervalized weighted finite automatons established in the process**, and **4) the results of the final adversarial attacks** have been opened.~~
- ~~The **experiment code**, the **pre-trained models** and the **crafted adversarial sets** are respectively uploaded in the ``Programs``, ``Models`` and ``UCR-Adv`` for reproducibility as well as to allow verification and possible improvement.~~

#### July 2022:
- ~~The sample program using TSFool to craft adversarial time series for an LSTM classifier in PowerCons Dataset from UCR Archive is added for reference in advance.~~
- ~~The work is in progress at present and the detailed description (as well as a possible technology paper) will be opened to the public soon.~~


## Core Idea

One of the possible explanations for the existence of the adversarial sample is that, the features of the input data cannot always fully and visually reflect the latent manifold, which makes it possible for samples that are considered to be similar in the external features to have radically different latent manifolds, and as a result, to be understood and processed in a different way by the DNN. Therefore, even a small perturbation in human cognition imposed on the correct sample may completely overturn the DNN's view of its latent manifold, so as to result in a completely different result. 

So if there is a kind of representation model that can simulate the way the DNN understands and processes input data, but distinguish different inputs by their original features in the high-dimensional space just like a human, then it can be used to capture the otherness between the latent manifold and external features of any input sample. And such otherness can serve as guidance to find the potentially vulnerable samples for the adversarial attack to improve its success rate, efficiency and quality.

In this project, the Interval Weighted Finite Automaton and Recurrent Neural Network (actually LSTM) are respectively the representation model and the DNN mentioned above. Further transferring this idea to other types of models and data is thought to be feasible tentatively, and such attempts are also in progress at present.


# Detailed Experiment Records

## The 10 Experimental Datasets from UCR Archive

**The UCR Time Series Classification Archive:** https://www.cs.ucr.edu/~eamonn/time_series_data_2018/

| ID     | Type      | Name                           | Train | Test | Class | Length |
|--------|-----------|--------------------------------|-------|------|-------|--------|
| CBF    | Simulated | CBF                            | 30    | 900  | 3     | 128    |
| DPOAG  | Image     | DistalPhalanxOutlineAgeGroup   | 400   | 139  | 3     | 80     |
| DPOC   | Image     | DistalPhalanxOutlineCorrect    | 600   | 276  | 2     | 80     |
| ECG200 | ECG       | ECG200                         | 100   | 100  | 2     | 96     |
| GP     | Motion    | GunPoint                       | 50    | 150  | 2     | 150    |
| IPD    | Sensor    | ItalyPowerDemand               | 67    | 1029 | 2     | 24     |
| MPOAG  | Image     | MiddlePhalanxOutlineAgeGroup   | 400   | 154  | 3     | 80     |
| MPOC   | Image     | MiddlePhalanxOutlineCorrect    | 600   | 291  | 2     | 80     |
| PPOAG  | Image     | ProximalPhalanxOutlineAgeGroup | 400   | 205  | 3     | 80     |
| PPOC   | Image     | ProximalPhalanxOutlineCorrect  | 600   | 291  | 2     | 80     |


## The Results of adversarial attack using TSFool and five common methods on the Experimental Datasets

### Exp. 1
#### - Dataset: CBF
#### - Original Model Acc (Test Set): 0.7511
| Method          | Attacked Acc | Generate Num | Time Cost (s) | Perturbation | CC         |
|-----------------|--------------|--------------|---------------|--------------|------------|
| FGSM            | 0.3311       | 900          | **0.004389**  | 14.12%       | 1.1481     |
| BIM             | 0.7022       | 900          | 0.029421      | 3.17%        | 0.9916     |
| DeepFool        | 0.2911       | 900          | 3.298845      | 12.29%       | 1.0994     |
| PGD             | 0.3311       | 900          | 0.029949      | 14.28%       | 1.1492     |
| Transfer Attack | 0.7422       | 900          | -             | **2.60%**    | 1.0105     |
| TSFool          | **0.2111**   | 720          | 0.042502      | 7.48%        | **0.7425** |

### Exp. 2
#### - Dataset: DPOAG
#### - Original Model Acc (Test Set): 0.7842
| Method          | Attacked Acc | Generate Num | Time Cost (s) | Perturbation | CC         |
|-----------------|--------------|--------------|---------------|--------------|------------|
| FGSM            | 0.4245       | 139          | **0.00319**   | 64.20%       | 1.1593     |
| BIM             | 0.7554       | 139          | 0.028328      | 13.59%       | 1.1577     |
| DeepFool        | 0.1727       | 139          | 0.961403      | 50.61%       | **1.1297** |
| PGD             | 0.3525       | 139          | 0.02906       | 64.68%       | 1.166      |
| Transfer Attack | 0.8575       | 400          | -             | 9.62%        | 1.8219     |
| TSFool          | **0.1071**   | 140          | 0.036396      | **4.93%**    | 1.6093     |

### Exp. 3
#### - Dataset: DPOC
#### - Original Model Acc (Test Set): 0.7319
| Method          | Attacked Acc | Generate Num | Time Cost (s) | Perturbation | CC         |
|-----------------|--------------|--------------|---------------|--------------|------------|
| FGSM            | 0.5833       | 276          | **0.002116**  | 42.21%       | **0.7966** |
| BIM             | 0.6413       | 276          | 0.022393      | 8.84%        | 0.9383     |
| DeepFool        | 0.442        | 276          | 1.974539      | 38.69%       | 0.9372     |
| PGD             | 0.5833       | 276          | 0.02313       | 42.54%       | 0.7967     |
| Transfer Attack | 0.6767       | 600          | -             | 5.24%        | 1.0938     |
| TSFool          | **0.0652**   | 660          | 0.011313      | **2.93%**    | 0.8381     |

### Exp. 4
#### - Dataset: ECG200
#### - Original Model Acc (Test Set): 0.7400
| Method          | Attacked Acc | Generate Num | Time Cost (s) | Perturbation | CC         |
|-----------------|--------------|--------------|---------------|--------------|------------|
| FGSM            | 0.58         | 100          | **0.003786**  | 21.07%       | 1.293      |
| BIM             | 0.65         | 100          | 0.035593      | **4.10%**    | 0.9787     |
| DeepFool        | **0.02**     | 100          | 0.24211       | 17.35%       | 1.2834     |
| PGD             | 0.52         | 100          | 0.037794      | 21.51%       | 1.3359     |
| Transfer Attack | 0.73         | 100          | -             | 4.26%        | 1.0422     |
| TSFool          | 0.0571       | 140          | 0.019018      | 4.41%        | **0.6291** |

### Exp. 5
#### - Dataset: GP
#### - Original Model Acc (Test Set): 0.9333
| Method          | Attacked Acc | Generate Num | Time Cost (s) | Perturbation | CC         |
|-----------------|--------------|--------------|---------------|--------------|------------|
| FGSM            | 0.4933       | 150          | **0.005317**  | 37.25%       | 1.8911     |
| BIM             | 0.9133       | 150          | 0.055787      | 8.61%        | 0.8701     |
| DeepFool        | 0.4467       | 150          | 4.224587      | 39.10%       | 1.7623     |
| PGD             | 0.4933       | 150          | 0.058562      | 37.50%       | 1.8903     |
| Transfer Attack | 0.9133       | 150          | -             | 5.06%        | 0.7851     |
| TSFool          | **0.0000**   | 80           | 0.19372       | **3.11%**    | **0.6815** |

### Exp. 6
#### - Dataset: IPD
#### - Original Model Acc (Test Set): 0.9650
| Method          | Attacked Acc | Generate Num | Time Cost (s) | Perturbation | CC         |
|-----------------|--------------|--------------|---------------|--------------|------------|
| FGSM            | 0.8571       | 1029         | **0.000281**  | 24.20%       | 0.933      |
| BIM             | 0.9368       | 1029         | 0.055788      | 3.90%        | 1.2255     |
| DeepFool        | **0.1535**   | 1029         | 0.09449       | 30.56%       | 1.114      |
| PGD             | 0.8192       | 1029         | 0.002599      | 24.59%       | 0.8952     |
| Transfer Attack | 0.932        | 1029         | -             | 4.98%        | 1.078      |
| TSFool          | 0.1588       | 340          | 0.008746      | **2.98%**    | **0.8021** |

### Exp. 7
#### - Dataset: MPOAG
#### - Original Model Acc (Test Set): 0.6429
| Method          | Attacked Acc | Generate Num | Time Cost (s) | Perturbation | CC         |
|-----------------|--------------|--------------|---------------|--------------|------------|
| FGSM            | 0.1883       | 154          | **0.006101**  | 123.79%      | 1.2606     |
| BIM             | 0.5519       | 154          | 0.055789      | 26.86%       | 1.0486     |
| DeepFool        | **0.013**    | 154          | 0.261087      | 80.59%       | 1.2047     |
| PGD             | 0.1883       | 154          | 0.061951      | 124.59%      | 1.2593     |
| Transfer Attack | 0.7925       | 400          | -             | 9.46%        | 1.4861     |
| TSFool          | 0.0333       | 60           | 0.09576       | **5.72%**    | **0.6831** |

### Exp. 8
#### - Dataset: MPOC
#### - Original Model Acc (Test Set): 0.6392
| Method          | Attacked Acc | Generate Num | Time Cost (s) | Perturbation | CC         |
|-----------------|--------------|--------------|---------------|--------------|------------|
| FGSM            | 0.5704       | 291          | **0.002051**  | 100.93%      | 1.0111     |
| BIM             | 0.6392       | 291          | 0.05579       | 21.82%       | 0.9903     |
| DeepFool        | 0.5361       | 291          | 2.326599      | 106.93%      | 0.9718     |
| PGD             | 0.5704       | 291          | 0.022377      | 101.61%      | 1.0123     |
| Transfer Attack | 0.4833       | 600          | -             | 9.35%        | 1.0294     |
| TSFool          | **0.1425**   | 400          | 0.018027      | **5.46%**    | **0.7291** |

### Exp. 9
#### - Dataset: PPOAG
#### - Original Model Acc (Test Set): 0.8976
| Method          | Attacked Acc | Generate Num | Time Cost (s) | Perturbation | CC         |
|-----------------|--------------|--------------|---------------|--------------|------------|
| FGSM            | 0.4878       | 205          | **0.00264**   | 134.01%      | **0.7961** |
| BIM             | 0.8878       | 205          | 0.055791      | 30.79%       | 1.5299     |
| DeepFool        | 0.1902       | 205          | 2.265096      | 121.79%      | 1.1564     |
| PGD             | 0.4878       | 205          | 0.027445      | 134.95%      | 0.7991     |
| Transfer Attack | 0.8829       | 205          | -             | 16.46%       | 1.6559     |
| TSFool          | **0.0643**   | 140          | 0.054581      | **15.74%**   | 0.899      |

### Exp. 10
#### - Dataset: PPOC
#### - Original Model Acc (Test Set): 0.7869
| Method          | Attacked Acc | Generate Num | Time Cost (s) | Perturbation | CC         |
|-----------------|--------------|--------------|---------------|--------------|------------|
| FGSM            | 0.6838       | 291          | **0.002175**  | 119.19%      | 1.7029     |
| BIM             | 0.732        | 291          | 0.055792      | 26.98%       | 1.0942     |
| DeepFool        | 0.3746       | 291          | 2.314018      | 96.88%       | **1.0729** |
| PGD             | 0.6838       | 291          | 0.024215      | 120.02%      | 1.7022     |
| Transfer Attack | 0.8419       | 291          | -             | 13.48%       | 1.212      |
| TSFool          | **0.1537**   | 540          | 0.015317      | **2.40%**    | 1.1282     |



<!-- ## The Experimental LSTM Classifiers -->

