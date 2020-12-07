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
  
The graphs of test accuracy and train loss are:  
<img src = "https://github.com/StephanieMussi/Text_Classification_CNN_RNN/blob/main/Figures/CharCNNAcc.png" width = 300 height = 200>
<img src = "https://github.com/StephanieMussi/Text_Classification_CNN_RNN/blob/main/Figures/CharCNNLoss.png" width = 300 height = 200>  

The train loss and test accuracy both converge early after about 25 epochs.  

## Word CNN Classfier
The architecture of the model is as below:  
<img src = "https://github.com/StephanieMussi/Text_Classification_CNN_RNN/blob/main/Figures/WordCNN.png" width = 500 height = 400>   

The codes for building this model are as below:  
```python
def __init__(self, vocab_size):
        super(WordCNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = layers.Embedding(vocab_size, EMBEDDING_SIZE, input_length=MAX_DOCUMENT_LENGTH)
        self.conv1 = layers.Conv2D(N_FILTERS, FILTER_SHAPE1, padding='VALID', activation='relu', use_bias=True, )
        self.pool1 = layers.MaxPool2D(POOLING_WINDOW, POOLING_STRIDE, padding='SAME')
        self.conv2 = layers.Conv2D(N_FILTERS, FILTER_SHAPE2, padding='VALID', activation='relu', use_bias=True)
        self.pool2 = layers.MaxPool2D(POOLING_WINDOW, POOLING_STRIDE, padding='SAME')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(MAX_LABEL, activation='softmax')
```  

After training, the accuracies are:  
|Train Accuracy|Test Accuracy|
|:-:|:-:|
|84.10% | 74.71%|  
  
The graphs of test accuracy and train loss are:  
<img src = "https://github.com/StephanieMussi/Text_Classification_CNN_RNN/blob/main/Figures/WordCNNAcc.png" width = 300 height = 200>
<img src = "https://github.com/StephanieMussi/Text_Classification_CNN_RNN/blob/main/Figures/WordCNNLoss.png" width = 300 height = 200>  

As can be seen, great fluctuation occur in both curves. The reason is the Adam optimizer, which reduces the time to convergence, but could result in fluctuation.    

## Character RNN Classifier
The architecture of the model is as below:  
<img src = "https://github.com/StephanieMussi/Text_Classification_CNN_RNN/blob/main/Figures/CharRNN.png" width = 200 height = 80>   

The implementation is as below:  
```python
def __init__(self, vocab_size, hidden_dim=10):
        super(CharRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.rnn = layers.RNN(tf.keras.layers.GRUCell(self.hidden_dim), unroll=True) # GRU
        self.dense = layers.Dense(MAX_LABEL, activation='softmax')
```  

The accuracies obtained are:  
|Train Accuracy|Test Accuracy|
|:-:|:-:|
|90.63%|69.71%|  
  
The graphs of test accuracy and train loss are:  
<img src = "https://github.com/StephanieMussi/Text_Classification_CNN_RNN/blob/main/Figures/CharRNNAcc.png" width = 300 height = 200>
<img src = "https://github.com/StephanieMussi/Text_Classification_CNN_RNN/blob/main/Figures/CharRNNLoss.png" width = 300 height = 200>  

As can be seen, the test accuracy converges early after about 30 epochs. However, the train loss does not seem to converge at the end.  


## Word RNN Classifier
The architecture of the model is as below:  
<img src = "https://github.com/StephanieMussi/Text_Classification_CNN_RNN/blob/main/Figures/WordRNN.png" width = 300 height = 100>   

The following codes are used to build the model:  
```python
def __init__(self, vocab_size, hidden_dim=10):
        super(WordRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding = layers.Embedding(vocab_size, EMBEDDING_SIZE, input_length=MAX_DOCUMENT_LENGTH)
        self.rnn = layers.RNN(tf.keras.layers.GRUCell(self.hidden_dim), unroll=True) # GRU
        self.dense = layers.Dense(MAX_LABEL, activation='softmax')
```  

The accuracies obtained are:  
|Train Accuracy|Test Accuracy|
|:-:|:-:|
|99.61%|85.00%|  
  
The graphs of test accuracy and train loss are:  
<img src = "https://github.com/StephanieMussi/Text_Classification_CNN_RNN/blob/main/Figures/WordRNNAcc.png" width = 300 height = 200>
<img src = "https://github.com/StephanieMussi/Text_Classification_CNN_RNN/blob/main/Figures/WordRNNLoss.png" width = 300 height = 200>   

## Comparision of The Models  
The test accuracy is used as the critiria, and is summarized in the table below:  
| |Char CNN	|Word CNN	|Char RNN	|Word RNN|
|:-:|:-:|:-:|:-:|:-:|
|Test Accuracy	|56.58%	|74.71%	|69.71%	|85.00%|  
  
The performance of classifying at word level is generally better than at character level. Besides, for both words and characters, RNN yields a higher accuracy than CNN.
However, the training time of RNN is significantly longer than CNN.  

## Add dropout  
After adding a dropout layer with drop rate of 50% before the output layer, the test accuracies are as below:  
|	|Char CNN	|Word CNN	|Char RNN	|Word RNN|
|:-:|:-:|:-:|:-:|:-:|
|W/out dropout	|56.58%	|74.71%	|69.71%	|85.00%|
|W/ dropout	|65.86%	|50.86%	|68.71%	|68.00%|  

From the table above, adding dropout helps to increase the test accuracy for Char CNN model which is overfitting. However, for the other models, dropout harms the performance.  


## Replace GRU Layer in RNN Models
### Vanilla RNN layer
* __Char RNN__   
  
After replacing the GRU layer with a Vanilla RNN layer, the accuracies are recorded, and graphs are plotted as below:  
|Train Accuracy|Test Accuracy|
|:-:|:-:|
|10.79%%|11.29%|  

    

<img src = "https://github.com/StephanieMussi/Text_Classification_CNN_RNN/blob/main/Figures/CharRNNVLoss.png" width = 300 height = 200>
<img src = "https://github.com/StephanieMussi/Text_Classification_CNN_RNN/blob/main/Figures/CharRNNVAcc.png" width = 300 height = 200>     

* __Word RNN__  
  
The accuracies obtained are:  
|Train Accuracy|Test Accuracy|
|:-:|:-:|
|8.05%|9.14%|  
  
The graphs of train loss and test accuracy are plotted:  
<img src = "https://github.com/StephanieMussi/Text_Classification_CNN_RNN/blob/main/Figures/WordRNNVLoss.png" width = 300 height = 200>
<img src = "https://github.com/StephanieMussi/Text_Classification_CNN_RNN/blob/main/Figures/WordRNNVAcc.png" width = 300 height = 200>   

It is clearly shown that Vanilla RNN layer leads to extremely fluatuating curves and low accuracies. This is caused by the Vanishing Gradient problem, where he gradients become smaller and smaller in the learning of long data sequences, and the updates become less meaningful, which means no real learning is done.  


### LSTM layer
* __Char RNN__   
  
The train and test accuracy are:  
|Train Accuracy|Test Accuracy|
|:-:|:-:|
|90.07%|70.29%|  
  
  

<img src = "https://github.com/StephanieMussi/Text_Classification_CNN_RNN/blob/main/Figures/CharRNNLLoss.png" width = 300 height = 200>
<img src = "https://github.com/StephanieMussi/Text_Classification_CNN_RNN/blob/main/Figures/CharRNNLAcc.png" width = 300 height = 200>   

The LSTM layer is free from Vanishing Gradient like Vanilla since it has forget gate, input gate and output gate to control the cell states. Actually, it leads to a higher accuracy than using a GRU layer.

* __Word RNN__    
  
The train and test accuracy are:  
|Train Accuracy|Test Accuracy|
|:-:|:-:|
|66/59%|36.14%|  
  
The graphs of train loss and test accuracy are plotted:  
<img src = "https://github.com/StephanieMussi/Text_Classification_CNN_RNN/blob/main/Figures/WordRNNLLoss.png" width = 300 height = 200>
<img src = "https://github.com/StephanieMussi/Text_Classification_CNN_RNN/blob/main/Figures/WordRNNLAcc.png" width = 300 height = 200>   

In this case, both curves does not converge at the end of 100 epochs and the final test accuracy is low. This is because the sentences are encoded as presences of words where the sequence is ignored, thus LSTM is not good at handing the data.

