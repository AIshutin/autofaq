import onnx

from onnx_tf.backend import prepare


if __name__ == "__main__":
    onnx_model = onnx.load("onnx/tmp/model.onnx")  # load onnx model
    tf_rep = prepare(onnx_model)  # prepare tf representation
    tf_rep.export_graph("tf/saved_model.pb")  # export the model
