# Tensorflow_Image_Captioning

Our Model comprises of two basic components:
1) Convolutional Neural Network: This is used for extracting the most important features of the input image, breaking down an image of size 224*224*3 into a vector of 1*4096 recognizable features.
2) Recurrent Neural Network: Input to this component of the model are the features, in which we implement LSTM cells, an improvised concept of RNN involving memory. Input to the cell at time step t will be the caption predicted at time step t-1 and the weights are accordingly adjusted to produce the most probable word keeping the embeddings into consideration.

![Model](/Images/Model.png "Model")

## Model

1) We take the image embedding from the VGG-16 model and use it to train the rest of our model.We pre-computed the 4,096 dimensional features to speed up training.
2) Writing them into a npy file, we make our work easier by using them to train the RNN.
3) To transform words into a fixed-length representation suitable for LSTM input, we use an embedding layer that learns to map words to 256 dimensional textual features (or word-embeddings). Word-embeddings help us represent our words as vectors, where similar word-vectors are semantically similar.
4) Our custom RNN model consists of a BasicLSTMCell implemented using Tensorflow, used to generate a word at every time step, looped till the maximum number of words that constitute a caption.
5) From the available captions, vocabulary is formed by setting a threshold on the frequency of the words. Each word is mapped into a 256 dimension vector. Also, the image is mapped into a word space to provide appropriate input to the LSTM cell.
6) To predict the next word, the embeddings of the previous word are passed to the LSTM and finally in the end these features are encoded back into words Caption is generated using a naive greedy approach.

## Math used

Using the Basic LSTM cell implemented by tensorflow, we make use of the following architecture:

![LSTM](/Images/LSTM_cell.png "LSTM")

Forget gate represented by ft <br>
Input gate represented by it <br>
Output gate represented by ot <br>

![Formula](/Images/Formula.png "Formula")

We use a batch size of 128 and number of units in LSTM cells 256. <br>
Weights that are learnt by our model - (Vocabulary size = 996 words) <br>
Word embedding(996 * 256), Word embedding bias (256 * 1) <br>
Image embedding(4096 * 256), Image embedding bias (256 * 1) <br>
Word encoding(256 * 996), Word encoding bias(996 * 1) <br>

## Model Description Continued

Activation Function: Softmax <br>
Loss Function Used: Cross Entropy <br>
Optimizer : Adam optimizer <br>
Testing : We currently have a model that gives the probability of a word appearing next in a caption, given the image and all previous words.An image is given as input to the model and iteratively outputs the next most probable word, building up a single caption. (Naive Greedy Search). <br>

Thus, the caption generator gives a useful framework for learning to map from images to human-level image captions. By training on large numbers of image-caption pairs, the model learns to capture relevant semantic information from visual features.


