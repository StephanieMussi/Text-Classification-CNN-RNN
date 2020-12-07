# Text_Classification_CNN_RNN
This project aims to perform text recognition tasks at word and character levels using Convolution Neural Network and Recurrent Neural Network.  
The training dataset in ["train_medium.csv"](https://github.com/StephanieMussi/Text_Classification_CNN_RNN/blob/main/train_medium.csv) contains 5600 paragraphs, each with one of 15 labels. The test set in ["test_medium.csv"](https://github.com/StephanieMussi/Text_Classification_CNN_RNN/blob/main/test_medium.csv) contains 700 samples for testing. 

## Character CNN Classifier
The architecture of the model is as below:  
<img src = "https://github.com/StephanieMussi/Text_Classification_CNN_RNN/blob/main/Figures/CharCNN.png" width = 700 height = 300>  
  
The codes of implementation are:  
```python
def __init__(self, vocab_size=256):
        super(CharCNN, self).__init__()
        self.vocab_size = vocab_size
        self.conv1 = layers.Conv2D(N_FILTERS, FILTER_SHAPE1, padding='VALID', activation='relu', use_bias=True)
        self.pool1 = layers.MaxPool2D(POOLING_WINDOW, POOLING_STRIDE, padding='SAME')
        self.conv2 = layers.Conv2D(N_FILTERS, FILTER_SHAPE2, padding='VALID', activation='relu', use_bias=True)
        self.pool2 = layers.MaxPool2D(POOLING_WINDOW, POOLING_STRIDE, padding='SAME')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(MAX_LABEL, activation='softmax')
```
  
The accuracies obtained from training for 100 epochs are:  
|Train Accuracy|Test Accuracy|
|:-:|:-:|
|100%|56.57%|  
  
The graphs of accuracy and loss are:  
<img src = "https://github.com/StephanieMussi/Text_Classification_CNN_RNN/blob/main/Figures/CharCNNAcc.png" width = 300 height = 200>
<img src = "https://github.com/StephanieMussi/Text_Classification_CNN_RNN/blob/main/Figures/CharCNNLoss.png" width = 300 height = 200>  



## Word CNN Classfier
The architecture of the model is as below:  
<img src = "https://github.com/StephanieMussi/Text_Classification_CNN_RNN/blob/main/Figures/WordCNN.png" width = 500 height = 400>   

## Character RNN Classifier
The architecture of the model is as below:  
<img src = "https://github.com/StephanieMussi/Text_Classification_CNN_RNN/blob/main/Figures/CharRNN.png" width = 200 height = 80>   

## Word RNN Classifier
The architecture of the model is as below:  
<img src = "https://github.com/StephanieMussi/Text_Classification_CNN_RNN/blob/main/Figures/WordRNN.png" width = 300 height = 100>   

