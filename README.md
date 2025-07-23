# NeuroAlign-SAM-for-multi-modal-MRI
A framework of hierarchical alignments, dual-domain cross-attention,mmscorecam. This code is the network implementation for paper $${\color{red} Leveraging \space multi-modal \space  MRI \space synergies  \space for  \space  cognitive  \space decline  \space  detection:  \space dual-domain \space cross-attention \space  enhanced \space by \space  hierarchical \space alignments}$$
## __Environment__  
```diff  
torch ==1.13.1
numpy == 1.22.3  
nibabel == 1.10.2  
torchcam == 0.3.2  
torchvision == 0.14.1  
einops == 0.6.0  
python == 3.9.0  
imageio == 2.31.1
``` 
## extract the imaging features
To run the model, you need to extract the sfc, dfc, and alff by Matlab. You can use the batch operation of spm12 to finish this. Then, you need to use Panda to generate FA. But you can also use FSL instead. Additionally, the input data shape might influence the kernel size of avgpooling, you need to change the kernel size, if has bugs.  

## run the model

### __Create k fold csv file__  
```diff
generate_csv.py
```
### train and validate the model 
```diff
train.py
```
### test the model 
```diff
test.py
```
## generate the avtivation map in multi-modal MRI  
### replace the py files in initial torchcam package with the files in $\color[rgb]{1,0,1} method$ folder in this respority, including
```diff
activation.py  
core.py
```
### run test.py  
```diff
test.py
```
### results
```diff
see the images folder
```
