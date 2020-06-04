# MultiChexNet

## Tabel of contents
- [Introduction](#Introduction)
- [Getting started](#Getting_started)
- [File Dataset ](#Dataset)

## Introduction
MultiChexNet is a model that is able to classify, detect and segment chest related disease in one forward step.
## Getting_started
You Can use this repo to train your independent model, whether it is classification,detection or segmenter only.
**You can find an example in the usage_example.py**
### steps:
1. Navigate to either the Classifier.py, Detector.py or segmenter.py and fill in the make model function    
2. Make an external file where you make an instance of the encoder and one of the following according to your task (classifier,detector, segmenter)
3. Call the ```ModelBlock.add_head(encoder, [classifier])``` to add your model head to the encoder 


## Dataset
