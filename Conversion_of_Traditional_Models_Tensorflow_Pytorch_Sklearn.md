# 💻Conversion of Traditional Models💻

Here is a walkthrough in how the conversion of the traditional machine learning models created by most common frame works such as  to ONNX format.

## 1. Conversion of Tensorflow Models to ONNX
___
To handle ONNX format conversion in Tensorflow there are two appraoch two follow:

### Converting Tensorflow Models to ONNX (tf2onnx)

To convert Tensorflow based Models to ONNX tf2onnx is the tool to do that.
you can easily install it using pip:
```bash
pip install tf2onnx
```
Here are a list of most common convert method available from tf2onnx:
| Convert_Method                   | Description                              |
|----------------------------------|------------------------------------------|
| `tf2onnx.convert.from_saved_model`  | Convert TensorFlow SavedModel format to ONNX.                  |
| `tf2onnx.convert.from_keras`        | Convert a Keras model to ONNX.                               |
| `tf2onnx.convert.from_graphdef`    | Convert TensorFlow GraphDef representation to ONNX.          |
| `tf2onnx.convert.from_function`     | Convert a TensorFlow computation graph represented by a Python function to ONNX.|
| `tf2onnx.convert.from_tensorflow`   | Convert TensorFlow model to ONNX.                            |

#### Example:
This is a simple example in Python that demonstrates how to convert a TensorFlow SavedModel to ONNX format using the `tf2onnx.convert.from_saved_model` method:

```python
import tensorflow as tf
import tf2onnx

# Assuming you have a SavedModel directory containing the TensorFlow model
saved_model_dir = '/path/to/saved_model_directory'

# Load the SavedModel using TensorFlow
loaded_model = tf.saved_model.load(saved_model_dir)

# Convert the SavedModel to ONNX format
onnx_model, _ = tf2onnx.convert.from_saved_model(saved_model_dir)

# Save the ONNX model to a file
onnx_model_path = '/path/to/output/model.onnx'
tf2onnx.save_model(onnx_model, onnx_model_path)

print("SavedModel converted to ONNX format successfully!")
```


### Loading ONNX model into  Tensorflow Model (onnx-tf)

To load ONNX model into Tensorflow you can onnx-tf package that has class `prepare` that is used to convert an ONNX model into a TensorFlow representation for inference and export.
you can install it using pip:
```bash
pip install onnx-tf
```

#### Example
here is a simple example in how to use onnx-tf to load model into tensorflow:

```python
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# Load the ONNX model
onnx_model = onnx.load("output/model.onnx")

# Prepare the ONNX model for TensorFlow
tf_rep = prepare(onnx_model)

# Export the TensorFlow model in SavedModel format
tf_saved_model_path = "path/to/your/saved_model"
tf_rep.export_graph(tf_saved_model_path, as_text=False)  # Set as_text=True to save in text format, if desired
```





To save a machine learning model to ONNX format, you can use the torch.onnx.export() function for PyTorch models or tools like onnxmltools for Scikit-learn models. Here's how to do it for a PyTorch model:




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







As of my last update in September 2021, the process of converting a TensorFlow SavedModel to ONNX using the tf2onnx library involved several steps. Please note that newer versions of libraries and tools might have been released since then, so I recommend checking the latest documentation for tf2onnx for any updates. Below are the general steps to perform the conversion:

1. Install tf2onnx:
Make sure you have tf2onnx installed on your system. You can install it via pip:

```bash
pip install tf2onnx
```

2. Load the TensorFlow SavedModel:
Load the TensorFlow SavedModel using the TensorFlow library:

```python
import tensorflow as tf

# Replace 'path/to/saved_model' with the actual path to your SavedModel directory
model = tf.saved_model.load('path/to/saved_model')
```

3. Convert the TensorFlow graph to ONNX format:
Use tf2onnx to convert the TensorFlow graph to ONNX format:

```python
import tf2onnx

# Convert the TensorFlow model to ONNX
onnx_model, _ = tf2onnx.convert.from_saved_model('path/to/saved_model', opset=12)
```

The `opset` parameter specifies the ONNX operator set version to use. It's recommended to use the latest available version (check the tf2onnx documentation for the latest supported opset version).

4. Save the ONNX model to a file:
Save the ONNX model to a file on your disk:

```python
# Replace 'path/to/output_model.onnx' with the desired output file path
with open('path/to/output_model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())
```

That's it! After these steps, you should have the ONNX model saved as a file, ready for use with other ONNX-compatible tools and frameworks.

Remember that TensorFlow and tf2onnx might receive updates, so it's a good idea to check for the latest documentation and potential changes in the conversion process. Additionally, not all TensorFlow operations can be directly translated to ONNX, so it's possible that some models may require additional adjustments during the conversion process.








onnx_model_path = "linear_regression.onnx"
onnxmltools.utils.save_model(onnx_model, onnx_model_path)
```








I apologize for the confusion earlier. To clarify, you want to use the `onnx_tf` library to load an ONNX model and then save it in TensorFlow's `SavedModel` format. Here's the code to achieve that:

```python
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# Load the ONNX model
onnx_model = onnx.load("output/model.onnx")

# Prepare the ONNX model for TensorFlow
tf_rep = prepare(onnx_model)

# Export the TensorFlow model in SavedModel format
tf_saved_model_path = "path/to/your/saved_model"
tf_rep.export_graph(tf_saved_model_path, as_text=False)  # Set as_text=True to save in text format, if desired
```

Replace `"output/model.onnx"` with the path to your ONNX model file. The code will load the ONNX model, convert it to TensorFlow format using the `onnx_tf.backend.prepare()`, and then save it as a TensorFlow SavedModel in the directory specified by `tf_saved_model_path`.

Please note that you'll need to install the `onnx` and `onnx-tf` packages to run this code:

```bash
pip install onnx onnx-tf
```

This code will utilize the `onnx_tf` library to handle the conversion from ONNX to TensorFlow format and then save it as a `SavedModel`.
