# 🚀 🔥 💻 UnifiedMLArtifactONNX 🚀 🔥 💻
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
![image](https://github.com/yousif4111/UnifiedMLArtifactONNX/assets/46527978/f420b5dc-09f1-4169-baf7-a4571b3a26db)


Storing ML/DL models in **ONNX** format offers:
- Interoperability across frameworks,
- Cross-platform deployment, and
- Optimized inference.

It enables collaboration, future-proofing, and access to pre-trained models in the **ONNX model zoo**, making it a versatile choice for ML model storage.

## Proposed Methodology Scenario Example:
---
Imagine you have trained a model on **TensorFlow**, and now you want to deploy it in a **PyTorch**-based infrastructure. Instead of resorting to traditional conversion methods involving model weight storage and reconstruction the model artifact in pytorch then load the weight. this approach just require you to save tf model as onnx then simply load it into Pytorch then immediatly start using the model.
![Blank diagram (1)](https://github.com/yousif4111/UnifiedMLArtifactONNX/assets/46527978/2f0f4682-4329-4e1a-831d-d4b38966d71e)


## File Structure

This repository contains the following files and directories:

- `convert_to_onnx.py`: Python script for converting machine learning models to ONNX format.
- `inference_on_onnx.py`: Python script for performing inference with ONNX models using ONNX Runtime.
- `optimize_model.py`: Python script demonstrating model optimization techniques during ONNX conversion.

- `requirements.txt`: Text file containing the required libraries for running the code examples.

- `data/`: Directory containing sample data used in the scripts.

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
| torchvision   | `pip install torchvision` |
| onnx          | `pip install onnx`       |
| onnxruntime   | `pip install onnxruntime`|
| optimum       | `pip install optimum`    |

## References

1. [Markdown Guide](https://www.markdownguide.org/)
2. [GitHub Markdown Cheatsheet](https://guides.github.com/pdfs/markdown-cheatsheet-online.pdf)
3. [CommonMark Specification](https://spec.commonmark.org/)












