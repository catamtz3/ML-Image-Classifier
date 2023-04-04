# ML-Image-Classifier
In this homework, we will implement a deep convolutional neural network (CNN) for image classification. We will use a customized dataset with 5 classes (i.e. face, airplane, dog, car, tree), and the dataset contains a thousand 30x30 color images per class. This dataset is selected from AffectNet, ImageNet, and CIFAR-10. This "toy" dataset is small enough to run on a CPU so that you can taste deep learning with limited resources.



Turn in your answers (in a report.pdf file) and code (the answer.py file) for your submission. We strongly advise you to review lecture notes and PyTorch tutorial before starting this homework.

For grading purpose, do not change anything in other files in the starter code, and train your model for 10 epochs for each question. The training and evaluation code are already finished for you in the main.py and helper.py files. The train.csv and valid.csv files define the location and label for each image, which are then loaded into deep learning models by the loader.py code.

Read the run function in helper.py to understand the logic of training process.

Your code will be tested in Python 3 and PyTorch under the Linux environment, and the installation steps can be found in the appendix. You should not import any other libraries in your code.

For each question, you need to report your results in four aspects:

How many trainable parameters are in the model? It is printed in the terminal.
What is the best training accuracy?
What is the best validation accuracy? Is it better than the ones in previous questions?
According to the training and validation accuracies, does the model overfit your training data?
Neural Network
3 points

We have created a simple neural network with one hidden layer (NN class in the answer.py file). Train the model by using the command python main.py --model NN.

Make sure you understand the NN class in the answer.py file before answering other questions.

Simple Convolution Neural Network
5 points

We have created a simple convolutional neural network with one hidden convolutional layer and one hidden fully-connected layer (SimpleCNN class in the answer.py file). Finish the forward function and train the model with the command python main.py --model SimpleCNN. Remember to add the ReLU activation into your forward function.

Color Normalization
2 points

One way to resolve various lightning conditions in input images is to normalize the color of images. For simplicity, let us use 0.5 as the mean and 0.5 as the standard deviation for each color channel. Implement the norm transformer variable in the answer.py file, and run python main.py --model SimpleCNN --transform norm.

Deep Convolutional Neural Network
10 points

CNNs with only one convolutional layer can only extract simple features from images, but deeper CNNs can extract more complex information. In this question, you need to complete the DeepCNN class in the answer.py file.

As shown in the following table, when given an array (e.g. [8, 16, 32, "pool"]), the DeepCNN should create a deep network with corresponding convolutional layers (i.e. 8-channel convLayer, 16-channel convLayer, 32-channel convLayer, max pooling layer), and then add a fully-connected layer after the last convolutional (or pooling) layer.

Layer	Output Size	Output Channels
Input	30 x 30	3
Conv	28 x 28	8
ReLU	28 x 28	8
Conv	26 x 26	16
ReLU	26 x 26	16
Conv	24 x 24	32
ReLU	24 x 24	32
Max Pool	12 x 12	32
Linear	5
You can assume the input data for this model is always a 3 × 30 × 30 PyTorch Tensor (which is a 30 × 30 RGB image). We will always use 2D convolutional layers with kernel size 3, stride 1, padding 0, dilation 1, and group 1; you should add a ReLU activation function after every convolutional layer. Similarly, we will always use max-pooling layers with kernel size 2, padding 0, and dilation 1. You can use nn.Sequential to make your code cleaner.

You need to reshape tensors to (b × p) size before feeding them to the first fully-connected layer, where b is the batch size and p is the length of feature vector. Make sure you calculated the tensor size correctly.

With a 3 hidden-layer CNN (i.e. [8, 16, 32, "pool"]), run python main.py -m DeepCNN --layers 8 16 32 pool --transform norm.

Data Augmentation
5 points

Now, we want to perform the random affine transformation and random horizontal flip to train images. Use the torchvision.transforms package to finish the variable aug transformer. In order to make the result comparible to previous questions, you should include color normalization (as implemented in question 3) after data augmentation. The answer to this question is extremely short (less than 5 lines). Run python main.py --model DeepCNN --layers 8 16 32 pool --transform aug.
