# 🚀  💻 UnifiedMLArtifactONNX 🚀  💻
***

@auther [Yousif Abdalla](https://github.com/yousif4111)

When creating ML models with frameworks like TensorFlow, scikit-learn, or PyTorch, they are often stored in unique formats for each framework. However, this diversity can lead to compatibility issues during deployment. Data scientists and ML engineers may face challenges in converting models between frameworks, causing potential delays in the deployment process.

I have create this repository to make a guide on quick effecient way to convert from one format to another using unified ML Artifact [ONNX](https://onnx.ai/)

## 🤖 ONNX 🤖
***
ONNX (Open Neural Network Exchange) is an open standard format to represent machine learning (ML) models built on various frameworks such as PyTorch, Tensorflow/Keras, scikit-learn. 

[ONNX Runtime](https://onnxruntime.ai/) is also an open source project that’s built on the ONNX standard. It’s an inference engine optimized to accelerate the inference process of models converted to the ONNX format across a wide range of operating systems, languages, and hardware platforms. 

### Why ONNX? 🤔
___
![image](https://github.com/yousif4111/UnifiedMLArtifactONNX/assets/46527978/1b2f05d3-0e92-4e6b-85cf-e2bb576e71c0)


```diff
Storing ML models in ONNX format offers:
+ Interoperability across frameworks
+ Cross-platform deployment
+ Optimized inference.
+ Enables Collaboration
+ Future-proofing
+ Versatile chocie for ML model storage.
```


## Proposed Methodology Scenario Example:
---
Imagine you have trained a model on **TensorFlow**, and now you want to deploy it in a **PyTorch**-based infrastructure. Instead of resorting to traditional conversion methods involving model weight storage and reconstruction the model artifact in pytorch then load the weight. this approach just require you to save tf model as onnx then simply load it into Pytorch then immediatly start using the model.
![Blank diagram (1)](https://github.com/yousif4111/UnifiedMLArtifactONNX/assets/46527978/2f0f4682-4329-4e1a-831d-d4b38966d71e)


## Repo Contents

This repository contains the following files and directories:

- `LLM2Onnx.py`: Python script for converting Large Langauge models to ONNX format developed in both framework tensorflow or pytorch.
  
- `Converstion_of_Traditional_Model`: Walkthrough on how to perform the conversoin of regular model to onnx and load them.
  
- `MMdnn_guide`: Guide on how to use the [MMdnn](https://github.com/microsoft/MMdnn) open source tool developed by Microsoft as Universal Converter.

- `requirements.txt`: Text file containing the required libraries for running the code examples.

- `README.md`: Markdown file containing the repository's documentation.

Feel free to explore the different files and directories to understand how to convert and use ONNX models effectively.




Requirements:
---
Here a list of Libraries used in this repository:

| Library       | Installation Command    |
|---------------|-------------------------|
| transformers  | `pip install transformers` |
| torch         | `pip install torch`       |
| tensorflow    | `pip install tensorflow`  |
| onnx          | `pip install onnx`       |
| onnxruntime   | `pip install onnxruntime`|
| optimum       | `pip install optimum`    |

## References

1. [TensorFlow to ONNX](https://onnxruntime.ai/docs/tutorials/tf-get-started.html)
2. [EXPORTING A MODEL FROM PYTORCH TO ONNX](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
3. [MMdnn](https://github.com/microsoft/MMdnn)
4. [Transformer Model Optimization Tool](https://onnxruntime.ai/docs/performance/transformers-optimization.html)













