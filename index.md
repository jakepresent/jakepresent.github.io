# Project Proposal

## Introduction and Background

Recent advances in computing architectures and machine learning, particularly in the area of deep learning, has allowed for the generation of realistic data. Generative Adversarial Networks (GANs) are a neural network architecture composed of two parts:

They have a Generator that attempts to map what is called a latent vector space, pulled from a set of random distributions like Gaussians, to the space in which data lives.

GANs also have a Discriminator, which maps from a vector space of the data distribution to a binary classification of whether or not the data given to the network is real or counterfeit.

The purpose of this architecture is to learn a Generator that can create data that looks like it is from the distribution of real data. The Generator and Discriminators play a min-max game in which they make each other better over time. The Generator tries to outsmart the Discriminator and vice versa. These models are trained in an unsupervised manner using the above described min-max game optimization. [1] [2]


## Problem Description

We want to leverage novel machine learning methods to generate realistic looking and aesthetically pleasing art from a dataset of examples. We would also like to be able to render artwork that encompasses certain parameters of a painting like the year, artist, color content, and style. 

## Methods

For our project, we want to use generative modeling, particularly generative adversarial networks (GANs), to render artwork. The GANs will have convolutional layers in the Generator and Discriminator to help map the spatial data of the paintings down to a latent representation, which will encode various visual features of the paintings.

The second part of our project will involve training a supervised model that can map a set of information about paintings (such as its region, artist, color content, and year of creation) to a vector in the latent space of the GAN’s generator network. This will allow us to render artwork that fits a certain set of desirable characteristics. Interesting visualizations like latent space interpolation of visual characteristics are also enabled by combining the supervised and unsupervised approaches above.

## Potential Results

Below is a set of paintings that were computer generated using a model called CycleGAN. [3] [4] We want to use similar methods to generate novel paintings that are aesthetically pleasing and fit certain parameters that we describe.

![](https://miro.medium.com/max/875/1*CX690BeurSxHFJPSiGW1ow.png)
[5]

## Discussion

This project explores many recent advancements in machine learning and generative modeling. It has a supervised and unsupervised component, and expands our knowledge of various areas of machine learning. The project fits within the scope of the class and is doable in the time span given. 


## References

- [Generative Adversarial Nets](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) [1]
    - Outlines general GAN approach and gives theoretical background

- [A Gentle Introduction to Generative Adversarial Networks (GANs)](https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/) [2]
    - Gives more general background and good information on GANs

- [GANGogh: Creating Art with GANs. Introduction](https://towardsdatascience.com/gangogh-creating-art-with-gans-8d087d8f74a1) [3]
    - Describes a novel project similar to ours

- [CycleGAN Paper](https://arxiv.org/abs/1703.10593) [4]
    - Details the CycleGAN method referenced above
- [Image Source](https://towardsdatascience.com/cyclegans-to-create-computer-generated-art-161082601709) [5]

