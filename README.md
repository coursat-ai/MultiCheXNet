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
https://www.kaggle.com/nih-chest-xrays/data

For clearance: csv of whole dataset = Data_Entry_2017.csv , csv of detection = BBox_List_2017.csv

Data manipualtion consists of 3 functions:
- Adjust_data: Merging of csv from detection and csv of the whole dataset and filling the null values so it takes 2 arguments
- Adjust_box : Correction of label missmatch, then sorting the images according to their image index, then we group the duplicates into 1 row (Because detection can have more than 1 value ie. more than 1 disease detected), and grouping is done with some arrangement of columns and concatenation them into string format
- dfcat2dfid : Takes the csv of the whole dataset and converts the labels into ID/Category

Notes found while inspecting the data. Detection labels doesnt have to match the whole dataset label, it can be less as shown in the notebook provided (you can check the image index 00000032_037.png)
