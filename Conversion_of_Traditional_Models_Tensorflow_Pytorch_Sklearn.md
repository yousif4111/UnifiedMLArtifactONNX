# 💻Conversion of Traditional Models💻

Here is a walkthrough in how the conversion of the traditional machine learning models created by most common frame works such as  to ONNX format.

## 1. Conversion of Tensorflow Models to ONNX


## 3. Conversion of Scikit-learn Models

ONNX was initially designed for neural networks and deep learning models. 
However, ONNX has evolved to support 🔄 other types of models beyond neural networks, making it more versatile and suitable for various machine learning tasks. 🤖💡

### Here is example of simple **Linear Regression:**
---
```python
import onnxmltools
from sklearn.linear_model import LinearRegression

# Create and train a Scikit-learn Linear Regression model
model = LinearRegression()
# (Assume X_train and y_train are the input features and labels)
model.fit(X_train, y_train)

# Convert the Scikit-learn model to ONNX format
onnx_model = onnxmltools.convert_sklearn(model, initial_types=[('input', 'float32', X_train.shape[1])])
onnx_model_path = "linear_regression.onnx"
onnxmltools.utils.save_model(onnx_model, onnx_model_path)
```


