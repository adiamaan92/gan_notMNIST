# gan_notMNIST
Generative Adversarial Networks and its variations tried on not MNIST dataset.
Unlike MNIST notMNIST data set has letters (A - J 10 classes) from various typographies.

This data set has lots of false labels and the letters themselves are much more complicated than the 
digits in MNIST. GAN's are trained on these to see the feasibility of letter reproduction.

Data set -> http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html.

Vanilla GAN -> https://arxiv.org/abs/1406.2661
Conditional GAN -> https://arxiv.org/abs/1411.1784
Deep Convolutional GAN -> https://arxiv.org/abs/1511.06434
  are tried on the data set.

The code can be run on a Google Cloud Platform using the shell script gcloud-notMNIST.sh (google sdk needed).



