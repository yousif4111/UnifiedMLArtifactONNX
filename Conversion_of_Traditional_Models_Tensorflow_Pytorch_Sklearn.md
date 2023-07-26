# 💻Conversion of Traditional Models💻

This is a walkthrough in how the conversion of the simple machine learning models created by some of the common frameworks such Tensorflow and pytorch to ONNX format.

## 1. 1️⃣📚Conversion of Tensorflow Models to & from ONNX📚1️⃣
___
The following is a guide through the process of converting TensorFlow models to ONNX format and loading ONNX models into TensorFlow. 

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

### Tensorflow_ONNX_References

1. [tensorflow-onnx](https://github.com/onnx/tensorflow-onnx)
2. [Convert TensorFlow model to ONNX](https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/tensorflow-convert-model)
3. [Tutorial: Import an ONNX Model into TensorFlow for Inference](https://thenewstack.io/tutorial-import-an-onnx-model-into-tensorflow-for-inference/)
___


## 2. 2️⃣🌟Conversion of Pytorch Models to & from ONNX🌟2️⃣
___


### Convert PyTorch Model to ONNX
The conversion of a PyTorch model to ONNX format can be done directly within PyTorch itself using `torch.onnx.export()`, without the need for any external libraries Unlike TensorFlow. 
However there is a catch, to exoprt a mdoel, when we call `torch.onnx.export()` function this will execute the model, recording a trace of waht operators are used to compute the outputs. Because `export` runs the model, we need to provide an input tensor `x`. The values in this can be random as long as it is the right type and size. 

#### Example:
In this example we export the model with an input of batch_size 1, but then specify the first dimension as dynamic in the dynamic_axes parameter in torch.onnx.export(). The exported model will thus accept inputs of size [batch_size, 1, 224, 224] where batch_size can be variable.

```python
# Input to the model
x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
torch_out = torch_model(x)

# Export the model
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
```

### Convert ONNX Model to Pytorch
Unlikey saving Pytorch model directly to onnx you can't load them to pytorch and there is no supported related methods to pytorch.
However it can be done throug using Unversal Conversion [MMdnn](https://github.com/microsoft/MMdnn)

You can install it using:
```bash
pip install mmdnn
```
