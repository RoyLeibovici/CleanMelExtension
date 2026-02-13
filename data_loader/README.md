### Data loaders

This directory contains two data loaders for the different datasets used in the project. 
1. `inference_dataloader.py` is used for the inference phase of the project. The default batch size is 1.
2. `SPencn_NSdns_RIRreal.py` is used for the training phase of the project. The user should replace the speech, noise and RIR paths with the paths to the respective datasets. 
3. To generate the synthetic data for training, run
```
# in root directory
python -m data_loader.SPencn_NSdns_RIRreal
````
<font color=gray> Hint: you could modify the hyperparameters by input arguments. Please refer to the `main()` function for more details. </font>

