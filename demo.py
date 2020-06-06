from Classifier import Classifier
from Encoder import Encoder
from ModelBlock import ModelBlock

encoder = Encoder()
classifier = Classifier(input_dim=(7,7,1024))

model=  ModelBlock.add_head(encoder, [classifier])

print(model.summary())