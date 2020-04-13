All datasets are in csv format where each row corresponds to one measurement.
Features are comma separated.
No pre-processing (except the feature selection for 50 features datasets) is done.
Separate csv file with labels.
The naming convention is train.csv and test.csv.
To select n measurements, take first n from the file.

--------

Currently:
test and train sets are not yet divided (as the data stems actually from from data set)
For each experiment/ scenario/ paper we may use different maximal training data sizes.

--------

svn structure:

- dataset:
  - AES Shivam
    - HW
    - Value
    - traces
  - DPAv2
    - HW
    - Value
    - traces
  - DPAv4
    - HW
    - Value
    - traces
  - Random Delay
    - HW
    - Value
    - traces
  - Masking
    - HW
    - Value
    - traces

--------

traces folders contain:
  - traces_complete.csv (without (or nearly without) pre-selection)
  - traces_50_HW.csv (50 feature preselected according to abs corr with HW model)
  - traces_50_value.csv (50 feature preselected according to abs corr with intermediate value model)

Value and HW folders contain one model.csv file

--------

Sample sizes:

AES Shivam: 100.000, 1250 features max (whole trace); maximal original data set available 500.000

DPAv4: 100.000, 3000 features

Random delay: 50.000, 3500 features max (whole trace); maximal original data set

Masked: TO BE ADDED

DPAv2: TO BE ADDED (if needed)
