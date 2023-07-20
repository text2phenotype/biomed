"""
Loosely based on 🤗Transformers convert_pytorch_checkpoint_to_tf2.py
https://github.com/huggingface/transformers/blob/master/src/transformers/convert_pytorch_checkpoint_to_tf2.py

Requires pytorch to be installed, which is not currently one of our dependencies

"""
import os
import shutil
import torch
import numpy as np
from transformers import BertConfig, TFBertForPreTraining, BertForPreTraining
from transformers import TFBertPreTrainedModel, BertPreTrainedModel

from transformers.modeling_tf_pytorch_utils import load_pytorch_checkpoint_in_tf2_model

WEIGHTS_NAME = "pytorch_model.bin"
TF2_WEIGHTS_NAME = "tf_model.h5"
CONFIG_NAME = "config.json"

COMPARE_MODELS = True

PT_BERT_FOLDER = "/Users/michaelpesavento/Desktop/pretrained_bert/pretrained_bert_tf/biobert_pretrain_output_all_notes_150000"
TF_OUT_PATH = "/Users/michaelpesavento/Desktop/pretrained_bert/tf_biobert_pretrain_output_all_notes_150000"


def convert_pt_to_tf(pytorch_checkpoint_path, tf_dump_path):
    model_class = TFBertForPreTraining
    pt_model_class = BertForPreTraining
    # model_class = TFBertPreTrainedModel
    # pt_model_class = BertPreTrainedModel

    # load config
    config_name = [name for name in os.listdir(pytorch_checkpoint_path) if name.endswith("config.json")][0]
    if not config_name:
        raise ValueError(f"Unable to find *config.json file in folder: {pytorch_checkpoint_path}")
    config = BertConfig.from_pretrained(os.path.join(pytorch_checkpoint_path, config_name))
    print("Building TensorFlow model from configuration: {}".format(str(config)))

    # The conversion script sets these, but we don't want them set True for our use case.
    # config.output_hidden_states = True
    # config.output_attentions = True

    # set up the TF model to have weights loaded
    tf_model = model_class(config)
    pytorch_checkpoint_filename = os.path.join(pytorch_checkpoint_path, WEIGHTS_NAME)
    tf_model = load_pytorch_checkpoint_in_tf2_model(tf_model, pytorch_checkpoint_filename)

    # sanity check that model outputs are comparable
    if COMPARE_MODELS:
        tfo = tf_model(tf_model.dummy_inputs, training=False)  # build the network

        state_dict = torch.load(pytorch_checkpoint_filename, map_location="cpu")
        pt_model = pt_model_class.from_pretrained(
            pretrained_model_name_or_path=None, config=config, state_dict=state_dict
        )

        with torch.no_grad():
            pto = pt_model(**pt_model.dummy_inputs)

        np_pt = pto[0].numpy()
        np_tf = tfo[0].numpy()
        diff = np.amax(np.abs(np_pt - np_tf))
        print("Max absolute difference between models outputs {}".format(diff))
        assert diff <= 2e-2, "Error, model absolute difference is >2e-2: {}".format(diff)

    # Save pytorch-model
    weights_path = os.path.join(tf_dump_path, TF2_WEIGHTS_NAME)
    print("Save TensorFlow model to {}".format(weights_path))
    tf_model.save_weights(weights_path, save_format="h5")
    config.save_pretrained(tf_dump_path)

    if "vocab.txt" in os.listdir(pytorch_checkpoint_path):
        shutil.copy(
            os.path.join(pytorch_checkpoint_path, "vocab.txt"),
            os.path.join(tf_dump_path, "vocab.txt"),
        )


if __name__ == "__main__":
    convert_pt_to_tf(PT_BERT_FOLDER, TF_OUT_PATH)
