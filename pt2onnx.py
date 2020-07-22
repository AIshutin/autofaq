from transformers.convert_graph_to_onnx import convert
from transformers import AutoTokenizer, DistilBertTokenizer
from os import environ
from psutil import cpu_count
from onnxruntime import InferenceSession, SessionOptions, get_all_providers
import os

def pt2onnx(model_dir):
    # Handles all the above steps for you
    new_dir = f"onnx/{os.path.basename(model_dir)}/"
    convert(framework="pt", model=model_dir,
                            output=os.path.join(new_dir, 'model.onnx'), opset=11)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.save_pretrained(new_dir)

def create_model_for_provider(model_path: str, provider: str) -> InferenceSession:
  assert provider in get_all_providers(), f"provider {provider} not found, {get_all_providers()}"
  # Few properties than might have an impact on performances (provided by MS)
  options = SessionOptions()
  options.intra_op_num_threads = 1

  # Load the model as a graph and prepare the CPU backend
  return InferenceSession(model_path, options, providers=[provider])

def prepare_input(tokens_ids):
    inputs_onnx = {k: v.cpu().detach().numpy() for k, v in tokens_ids.items()}
    print(inputs_onnx)
    return inputs_onnx

def load_model(model_dir):
    print(f"model dir is {model_dir}")
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)

    # Constants from the performance optimization available in onnxruntime
    # It needs to be done before importing onnxruntime
    environ["OMP_NUM_THREADS"] = str(cpu_count(logical=True))
    environ["OMP_WAIT_POLICY"] = 'ACTIVE'

    cpu_model = create_model_for_provider(os.path.join(model_dir, "model.onnx"), "CPUExecutionProvider")

    # Inputs are provided through numpy array
    model_inputs = tokenizer("My name is Bert", return_tensors="pt")
    inputs_onnx = prepare_input(model_inputs)

    # Run the model (None = get all the outputs)
    cpu_model.run(None, inputs_onnx)[0]

    return cpu_model

if __name__ == "__main__":
    from transformers import TFAutoModel, AutoTokenizer, AutoModel
    model = AutoModel.from_pretrained('distilbert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    dir = 'tmp'
    os.system(f'rm -r {dir}')
    os.system(f'rm -r onnx/{dir}')
    os.system(f'mkdir {dir}')

    model.save_pretrained(dir)
    tokenizer.save_pretrained(dir)

    pt2onnx(dir)
    load_model(os.path.join('onnx', dir))
