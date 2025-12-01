# Region-Wise Brain Age Estimation

<img width="2796" height="1010" alt="image" src="https://github.com/user-attachments/assets/635a265f-907c-4fb1-90f0-728228cf6a8f" />

### Relevant Papers
- Investigation of Genetic Variants and Causal Biomarkers Associated with Brain Aging Jangho Kim, Junhyeong Lee, Seunggeun Lee medRxiv 2022.03.04.22271813; doi: https://doi.org/10.1101/2022.03.04.22271813
-----------------------------


### Preprocessing: Region Extraction
- To preprocess data for this project, use the `preprocessing.py` script located in the `src` directory. This script is capable of running in the background and saving the output logs to a file.
- Run the script with specified start and end indices, and a trial number. 
```
nohup python src/preprocessing.py --start_index 5001 --end_index 10000 --trial 1 > log/prc/prc_10000.out 2>&1 &
```
-----------------------------

### Model Training using CNN
Run the script with specified region of interest.
```
nohup python src/main_cv.py --roi 3 --dataset 0 --epochs 40 --batch_size 4 --model_load 1 > log/train/tf_temporal_lobe.out 2>&1 &
```
