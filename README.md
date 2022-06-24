### PyTorch implementation of DeepER


### Running a model:

     CUDA_VISIBLE_DEVICES=1 python main.py --algorithm deeper --dataset WN18RR --mode uni
     CUDA_VISIBLE_DEVICES=1 python main.py --algorithm mdeeper --dataset FB-IMG --mode mul


### Hyperparameters: 

Available in hyperparameters.txt 


### Requirements

	PyTorch	1.8.1
	GrouPy

Access GrouPy (Pytorch) at https://github.com/adambielski/GrouPy 

Unzip data.rar

Access FB-IMG, WN9-IMG, and their pretrained embeddings at https://github.com/UKPLab/starsem18-multimodalKB 	

### Acknowledgment: 
DeepER is implemented in the code base taken from https://github.com/ibalazevic/HypER
