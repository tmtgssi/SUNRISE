import os
 
path1='./classical_SR_datasets/DIV2K_train_LR_bicubic/'
path2='./classical_SR_datasets/DIV2K_train_LR_bicubic_renamed/'
for name in os.listdir(path1):
	source = path1+name
	dest = path2 +name[:-6]+'.png'
	os.rename(source, dest)
	
path3='./classical_SR_datasets/DIV2K_valid_LR_bicubic/'
path4='./classical_SR_datasets/DIV2K_valid_LR_bicubic_renamed/'
for name in os.listdir(path3):
	source = path3+name
	dest = path4 +name[:-6]+'.png'
	os.rename(source, dest)
