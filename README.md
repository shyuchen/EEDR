#  Enhancing Embedding Diversity and Robustness for Image-Text Retrieval in Remote Sensing

![image](https://github.com/ycharlene/EEDR/blob/main/images/Figure%202.jpg)

## Usage 
### requirements
we use single RTX A6000 48G GPU for training and evaluation
```
torch 1.9.0
torchvision 0.10.0
numpy 1.21.6
```
### Prepare Datasets
Download RSITMD dataset from [here](https://paperswithcode.com/dataset/rsitmd), and RSICD dataset from [here](https://paperswithcode.com/dataset/rsicd).

Organize them in `your_dataset_root_dir`folder as follows:
```
|-- your_dataset_root_dir/
|  |-- <RSICD>/
|     |-- images
|     |-- train_caps.txt
|     |-- train_caps_verify.txt
|     |-- train_filename.txt
|     |-- val_caps_verify.txt
|     |-- val_filename_verify.txt
|
|  |--<RSITMD>/
|     |--images

```

## Training
```
python train.py
```

## Testing
```
python test.py
```

## Acknowledgments
Some components of this code implementation are adopted from [IRRA](https://github.com/anosorae/IRRA). We sincerely appreciate for their contributions.
