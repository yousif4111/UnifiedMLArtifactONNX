# :electron::dependabot:Model Management deep neural network(MMdnn) Guide:dependabot::electron:
@Authoer [Yousif](https://github.com/yousif4111)

This walkthrough will demonstrate how to use MMdnn as a conversion tool for deep neural networks.

## What is MMdnn?:neutral_face:

MMdnn (Model Management deep neural network) is an open-source library designed to help users easily convert, visualize, and deploy deep learning models.
It supports various popular deep learning frameworks such as TensorFlow, PyTorch, Keras, Caffe, and more, making it a versatile tool for model conversion.

## Installation
You can install it directly through the following command:
```bash
pip install mmdnn
```

## 🎲🍀1️⃣Example_1: Case pre-train Tensorflow model1️⃣🍀🎲
___
First Example is guide on how to use MMdnn on **TensorFlow** is a pre-trained object detection model `FRCNN.pb` based on the Faster R-CNN architecture
to convert it into 

Project Directory Structure:


   ```
   models/
     ├── vgg16.savedModel
     └── vgg16.pt
   ```


