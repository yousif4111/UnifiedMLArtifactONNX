# -------------------------------------------------------------------------
#   Script to convert common LLM to ONNX 
#    @Author Yousif
# --------------------------------------------------------------------------

# -------------------------------------------- How to run ------------------------------------------------------- 
#        python LLM2Onnx.py --model_dir /path/to/model --output /path/to/onnx_output --framework pt --optim
# -------------------------------------------- How to run -------------------------------------------------------
import os
import json
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from transformers.convert_graph_to_onnx import convert
from onnxruntime.transformers import optimizer
import subprocess
import argparse


parser = argparse.ArgumentParser(description="Convert a pre-trained model to ONNX format and optionally optimize it.")
parser.add_argument("--model_dir", type=str, required=True, help="Path to the model directory")
parser.add_argument("--output", type=str, required=True, help="Output directory for the ONNX model")
parser.add_argument("--framework", type=str, default="pt", help="The framework used to develop the model (default: pt)")
parser.add_argument("--optim", action="store_true", help="Whether to perform ONNX optimization (default: False)")

args = parser.parse_args()





MODEL_TYPES_LIST = ["bart","bert","bert_tf","bert_keras","clip","gpt2","gpt2_tf","gpt_neox","swin","tnlr","t5","unet","vae","vit"]

def get_model_from_directory(model_dir):

    # --------------------------------------------------
    # Find model type and chekc if it can be converted
    # --------------------------------------------------
    # Check if the directory exists
    if not os.path.exists(model_dir):
        print(f"Error: Directory '{model_dir}' does not exist.")
        return None

    # Load the config.json file
    config_file = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_file):
        print(f"Error: 'config.json' file not found in the directory '{model_dir}'.")
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
        # ---------------------------
        # Secont to load the model
        # ---------------------------
        model = AutoModel.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Get the model type and task from the model configuration
        num_heads = model.config.num_attention_heads
        hidden_size = model.config.hidden_size

        return model,tokenizer,model_type,num_heads,hidden_size
    else:
        print(f"The model Detected is from type: {model_type}")
        return None

# argparse list
model_dir = args.model_dir #"model_dir"
output = args.output #"onnx_dir"
frame_work = args.framework #"pt"
optim = args.optim
#====================

try:
    model, tokenizer, model_type, num_heads, hidden_size = get_model_from_directory(model_dir)
except Exception:
    print("Unfortunately, the model is not supported yet")
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


