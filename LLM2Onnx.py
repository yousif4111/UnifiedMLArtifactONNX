# -------------------------------------------------------------------------
#   Script to convert common LLM to ONNX 
#    @Author Yousif
# --------------------------------------------------------------------------

# -------------------------------------------- How to run ------------------------------------------------------- 
#        python LLM2Onnx.py --model_dir /path/to/model --output /path/to/onnx_output --framework pt --optim
# -------------------------------------------- How to run -------------------------------------------------------
import argparse


parser = argparse.ArgumentParser(description="Convert a pre-trained model to ONNX format and optionally optimize it.")
parser.add_argument("--model_dir", type=str, required=True, help="Path to the model directory")
parser.add_argument("--output", type=str, required=True, help="Output directory for the ONNX model")
parser.add_argument("--framework", type=str, default="pt", help="The framework used to develop the model (default: pt)")
parser.add_argument("--optim", action="store_true", help="Whether to perform ONNX optimization (default: False)")

args = parser.parse_args()

import os
import json
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from transformers.convert_graph_to_onnx import convert
from onnxruntime.transformers import optimizer
import subprocess





MODEL_TYPES_LIST = ["bart","bert","bert_tf","bert_keras","clip","gpt2","gpt2_tf","gpt_neox","swin","tnlr","t5","unet","vae","vit"]

def get_model_from_directory(model_dir):
    """
    Description:
    This function loads a pre-trained model from the specified 'model_dir'
    and its corresponding tokenizer. It reads the model configuration
    from the 'config.json' file in the model directory to determine the model type.
    If the model type is supported (present in MODEL_TYPES_LIST), it returns the
    loaded model, tokenizer, model type, number of attention heads, and hidden size.

    Args:
        model_dir (str): The path to the directory containing the pre-trained model.
        
    Outputs:
        model (transformers.PreTrainedModel): The loaded pre-trained model.
        tokenizer (transformers.PreTrainedTokenizer): The corresponding tokenizer for the model.
        model_type (str): The type of the pre-trained model detected.
        num_heads (int): The number of attention heads in the model's configuration.
        hidden_size (int): The hidden size of the model's configuration.
        
        If the model type is not supported (not present in MODEL_TYPES_LIST),
        it prints an error message and returns None.
    """

    # --------------------------------------------------
    # Find model type and check if it can be converted
    # --------------------------------------------------
    # Check if the directory exists
    if not os.path.exists(model_dir):
        print(f"\nError: Directory '{model_dir}' does not exist.\n")
        return None

    # Load the config.json file
    config_file = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_file):
        print(f"\nError: 'config.json' file not found in the directory '{model_dir}'.\n")
        return None

    # Read the model configuration from the 'config.json' file
    with open(config_file, "r") as f:
        config_data = json.load(f)

    # Extract the model type from the configuration
    model_type = config_data.get("model_type", None)
    print(f"""
        =================== A Model has been detected succesfuuly ========================
                        The model detected is from type: {model_type}
        =================== A Model has been detected succesfuuly ========================
    """)

    if model_type in MODEL_TYPES_LIST:
        # ---------------------------------------------------------
        # Secont to load the model and find num_head & hidden_size
        # ---------------------------------------------------------
        model = AutoModel.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Get the model type and task from the model configuration
        num_heads = model.config.num_attention_heads
        hidden_size = model.config.hidden_size

        return model,tokenizer,model_type,num_heads,hidden_size
    else:
        print("\n====================================================")
        print(f"The model Detected is from type: {model_type}")
        print("Unfortunately, the model is not supported yet. Yousif still hustling... :)\n")
        return None

# argparse list
model_dir = args.model_dir #"model_dir"
output = args.output #"onnx_dir"
frame_work = args.framework #"pt or tf"
optim = args.optim
#====================

try:
    model, tokenizer, model_type, num_heads, hidden_size = get_model_from_directory(model_dir)
except Exception:
    exit()


#-------------------------
# creating new directory
#-------------------------
# output="onnx_dir"
output=output+f"/onnx_{model_type}"
clear_dir = f"rm -rf {output}"
create_dir= f"mkdir {output}"

# Execute the command using subprocess
try:
    subprocess.run(clear_dir, shell=True, check=True)
    subprocess.run(create_dir, shell=True, check=True)
    print("Directory sucessfully created")
except subprocess.CalledProcessError as e:
    print(f"Error while creating directory: {e}")
    exit()




frame_work="pt"
path_output=Path(f"{output}/{model_type}.onnx")
# Handles all the above steps for you
convert(framework=frame_work, model=model_dir, output=Path(f"{output}/{model_type}.onnx"), opset=11)


#--------------------
# Optimization Part
#-------------------
def llm_onnx_optimization(output,model_type,num_heads,hidden_size):
    """
    Description:
    This function performs optimization on the ONNX model file located at 'output' directory,
    using the specified 'model_type', 'num_heads', and 'hidden_size'. It converts the model's
    floating-point parameters to 16-bit floating-point (fp16) format to reduce memory usage
    and improve performance on devices with support for reduced precision computations.
    
    Args:
        output (str): The directory path where the original ONNX model is located.
        model_type (str): The type of the model used for optimization (e.g., 'bert', 'gpt2').
        num_heads (int): The number of attention heads in the model's configuration.
        hidden_size (int): The hidden size of the model's configuration.
        
    Outputs:
        None
        
    The function saves the optimized ONNX model to a new file with the name '{model_type}_optim_fp16.onnx'
    in the same directory as the original model. If the optimization process encounters an error,
    it may raise exceptions or print error messages, but it will not return any value.
    """
    path_output_optim = Path(f"{output}/{model_type}_optim_fp16.onnx")
    optimized_model = optimizer.optimize_model(str(path_output), model_type=model_type, num_heads=num_heads, hidden_size=hidden_size)
    optimized_model.convert_float_to_float16()
    optimized_model.save_model_to_file(str(path_output_optim))


if optim:
    print("""
    \n    ================ Otptimization in progress ================
          This will take few minutes depending no model size
    ================ Otptimization in progress ================        
    """)
    llm_onnx_optimization(output,model_type,num_heads,hidden_size)
    print("""
    The onnx Model has been Optimized Sucessfully !!
    """)
else:
    print("          The Model Has been Converted to Onnx without Optimization Sucessfully !!")


