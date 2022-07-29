from black import main
import torch
import os
from transformers.modeling_utils import prune_linear_layer
from transformers import AutoConfig, hubertForSequenceClassification
from transformers.file_utils import hf_bucket_url, cached_path

from .utils import calculate_parameters


def edit_config(config, additional_args):
    config.transform_embedding = additional_args.transform_embedding
    config.do_distill = additional_args.do_distill
    config.do_layer_distill = additional_args.do_layer_distill


def initialize_layer_transformation(model):
    model.layer_transformation.weight.data.copy_(
        torch.eye(len(model.layer_transformation.weight)))
    model.layer_transformation.bias.data.fill_(0)


def load_model_with_zs(model_path, model_class, zs=None):
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        config = AutoConfig.from_pretrained(model_path)
    model = model_class.from_pretrained(model_path, config=config)
    loaded_weights = torch.load(model_path, map_location="cpu")
    model.load_state_dict(loaded_weights)
    print(f"Load weights from {model_path}")
    update_params(model, zs)
    print(f"Model Size before pruning: {calculate_parameters(model)}")
    prune_model_with_z(zs, model)
    print(f"Model Size after pruning: {calculate_parameters(model)}")
    return model


def load_model(model_path, model_class, zs=None, num_labels=2):
    if zs is not None:
        model = load_model_with_zs(model_path, model_class, zs)
    else:
        # only and task name and model weights are accessible
        model = load_pruned_model(model_path, model_class)
        print(f"Model Size: {calculate_parameters(model)}")
    return model


# load the l0 module
def load_l0_module(model_path):
    l0_module_path = os.path.join(model_path, "l0_module.pt")
    if os.path.exists(l0_module_path):
        return torch.load(l0_module_path, map_location=torch.device('cpu'))
    else:
        return None


# z values could be in [0, 1), we update the parameters accordingly with z values
def update_params(model, zs):
    hubert = model.w2v_model

    config = model.config
    hidden_dims = config.hidden_size
    num_heads = config.num_attention_heads
    dims_per_head = hidden_dims // num_heads
    num_layers = config.num_hidden_layers

    if zs is not None:
        if "intermediate_z" in zs:
            for layer in range(num_layers):
                intermediate_z = zs["intermediate_z"][layer].cpu().squeeze(
                ).clone()
                hubert.encoder.layer[
                    layer].output.dense.weight.data = hubert.encoder.layer[
                        layer].output.dense.weight.data.mul(intermediate_z)
                if "mlp_z" in zs:
                    mlp_z = zs["mlp_z"][layer].cpu()
                    hubert.encoder.layer[
                        layer].output.dense.weight.data = hubert.encoder.layer[
                            layer].output.dense.weight.data.transpose(
                                0, 1).mul(mlp_z).transpose(0, 1)
                    hubert.encoder.layer[
                        layer].output.dense.bias.data = hubert.encoder.layer[
                            layer].output.dense.bias.data.mul(mlp_z)

        if "head_z" in zs:
            for layer in range(num_layers):
                head_z = zs["head_z"][layer].cpu().squeeze().clone()
                head_z = torch.repeat_interleave(head_z, dims_per_head)
                hubert.encoder.layer[
                    layer].attention.self.value.weight.data = hubert.encoder.layer[
                        layer].attention.self.value.weight.transpose(
                            0, 1).data.mul(head_z).transpose(0, 1)
                hubert.encoder.layer[
                    layer].attention.self.value.bias.data = hubert.encoder.layer[
                        layer].attention.self.value.bias.data.mul(head_z)
                if "head_layer_z" in zs:
                    head_layer_z = zs["head_layer_z"][layer].cpu()
                    hubert.encoder.layer[
                        layer].attention.output.dense.weight.data = hubert.encoder.layer[
                            layer].attention.output.dense.weight.transpose(
                                0, 1).data.mul(head_layer_z).transpose(0, 1)
                    hubert.encoder.layer[
                        layer].attention.output.dense.bias.data = hubert.encoder.layer[
                            layer].attention.output.dense.bias.data.mul(
                                head_layer_z)

        if "hidden_z" in zs:
            hidden_z = zs["hidden_z"].cpu().squeeze().clone()
            hubert.post_proj.weight.data =\
                hubert.post_proj.weight.data.mul(hidden_z)
            hubert.post_proj.bias.data = \
                hubert.post_proj.bias.data.mul(hidden_z)
            for layer in range(num_layers):
                hubert.encoder.layer[
                    layer].attention.self.key.weight.data = hubert.encoder.layer[
                        layer].attention.self.key.weight.data.mul(hidden_z)
                hubert.encoder.layer[
                    layer].attention.self.query.weight.data = hubert.encoder.layer[
                        layer].attention.self.query.weight.data.mul(hidden_z)
                hubert.encoder.layer[
                    layer].attention.self.value.weight.data = hubert.encoder.layer[
                        layer].attention.self.value.weight.data.mul(hidden_z)
                hubert.encoder.layer[
                    layer].attention.output.dense.weight.data = hubert.encoder.layer[
                        layer].attention.output.dense.weight.data.transpose(
                            0, 1).mul(hidden_z).transpose(0, 1)
                hubert.encoder.layer[
                    layer].attention.output.dense.bias.data = hubert.encoder.layer[
                        layer].attention.output.dense.bias.data.mul(hidden_z)
                hubert.encoder.layer[
                    layer].intermediate.dense.weight.data = hubert.encoder.layer[
                        layer].intermediate.dense.weight.data.mul(hidden_z)
                hubert.encoder.layer[
                    layer].output.dense.weight.data = hubert.encoder.layer[
                        layer].output.dense.weight.data.transpose(
                            0, 1).mul(hidden_z).transpose(0, 1)
            if hasattr(hubert.pooler, "dense"):
                hubert.pooler.dense.weight.data = hubert.pooler.dense.weight.data.mul(
                    hidden_z)


def prune_model_with_z(zs, model):
    if zs is None:
        return None, None
    hubert = model.w2v_model
    if "head_z" in zs:
        head_z = zs.get("head_z", None)
        head_layer_z = zs.get("head_layer_z", None)
        prune_heads = {}
        for layer in range(len(head_z)):
            head_z_layer = head_z[layer].cpu().squeeze().clone()
            if head_layer_z is not None:
                head_z_layer *= head_layer_z[layer]
            index = torch.where(head_z_layer == 0)[0].tolist()
            prune_heads[layer] = index
            print(
                f"Layer {layer}, heads {' '.join([str(i) for i in index])} pruned."
            )
        model.prune_heads(prune_heads)

    kept_intermediate_dims = None
    if "intermediate_z" in zs:
        kept_intermediate_dims = {}
        intermediate_zs = zs["intermediate_z"]
        mlp_z = zs.get("mlp_z", None)
        for layer in range(len(intermediate_zs)):
            intermediate_z_layer = intermediate_zs[layer].squeeze()
            intermediate_z_layer = intermediate_z_layer.cpu().clone()
            if mlp_z is not None:
                intermediate_z_layer *= mlp_z[layer]
            kept_intermediate_dims[layer] = intermediate_z_layer.nonzero(
            ).reshape(-1).tolist()

    def prune_layer_norm(layernorm, index):
        layernorm.weight = torch.nn.parameter.Parameter(
            layernorm.weight.index_select(0, index))
        layernorm.bias = torch.nn.parameter.Parameter(
            layernorm.bias.index_select(0, index))
        layernorm.normalized_shape = (len(index), )

    def prune_layer(layer, index, dim):
        layer = prune_linear_layer(layer, index, dim=dim)
        return layer

    if "hidden_z" in zs:
        hidden_zs = zs["hidden_z"]
        index = torch.LongTensor(
            hidden_zs.squeeze().nonzero().squeeze().tolist())
        index = index.to(model.device)

        hubert.post_proj.weight = torch.nn.parameter.Parameter(
            hubert.post_proj.weight.index_select(1, index).clone().detach())
        hubert.post_proj.bias = torch.nn.parameter.Parameter(
            hubert.post_proj.bias.index_select(1, index).clone().detach())

        for layer in range(0, 12):
            if hubert.encoder.layer[layer].attention.self.query is not None:
                hubert.encoder.layer[layer].attention.self.query = \
                    prune_layer(hubert.encoder.layer[layer].attention.self.query , index, dim=1)
                hubert.encoder.layer[layer].attention.self.key = \
                    prune_layer(hubert.encoder.layer[layer].attention.self.key , index, dim=1)
            if hubert.encoder.layer[layer].attention.self.value is not None:
                hubert.encoder.layer[layer].attention.self.value = \
                    prune_layer(hubert.encoder.layer[layer].attention.self.value , index, dim=1)
                hubert.encoder.layer[layer].attention.output.dense = \
                    prune_layer(hubert.encoder.layer[layer].attention.output.dense , index, dim=0)
                prune_layer_norm(
                    hubert.encoder.layer[layer].attention.output.LayerNorm,
                    index)
            if hubert.encoder.layer[layer].intermediate.dense is not None:
                hubert.encoder.layer[layer].intermediate.dense = \
                    prune_layer( hubert.encoder.layer[layer].intermediate.dense, index, dim=1)
                hubert.encoder.layer[layer].output.dense = \
                    prune_layer( hubert.encoder.layer[layer].output.dense, index, dim=0)
                prune_layer_norm(hubert.encoder.layer[layer].output.LayerNorm,
                                 index)

        # accommodate for different models
        if hasattr(model, "proj"):
            model.proj = prune_linear_layer(model.proj, index, dim=1)
        if hasattr(hubert.pooler, "dense"):
            hubert.pooler.dense = prune_linear_layer(hubert.pooler.dense,
                                                     index,
                                                     dim=1)
        if getattr(model, "layer_transformation", None) is not None:
            model.layer_transformation = prune_linear_layer(
                model.layer_transformation, index, dim=1)
            print("layer transformation",
                  model.layer_transformation.weight.shape)

    if kept_intermediate_dims is not None:
        prune_intermediate_layers(model, kept_intermediate_dims)

    # print model params
    for layer in range(0, 12):
        print("Layer:", layer)
        if hubert.encoder.layer[layer].attention.self.query is not None:
            print(
                "query:",
                hubert.encoder.layer[layer].attention.self.query.weight.shape)
            print("key:",
                  hubert.encoder.layer[layer].attention.self.key.weight.shape)
        else:
            print("query:", None)
            print("key:", None)
        if hubert.encoder.layer[layer].attention.self.value is not None:
            print(
                "value:",
                hubert.encoder.layer[layer].attention.self.value.weight.shape)
            print(
                "output:", hubert.encoder.layer[layer].attention.output.dense.
                weight.shape)
        else:
            print("value:", None)
            print("output:", None)
        if hubert.encoder.layer[layer].intermediate.dense is not None:
            print("up:",
                  hubert.encoder.layer[layer].intermediate.dense.weight.shape)
            print("down:",
                  hubert.encoder.layer[layer].output.dense.weight.shape)
        else:
            print("up", None)
            print("down", None)


def prune_intermediate_layers(model, keep_dims):
    hubert = model.w2v_model
    device = model.device
    for layer in keep_dims:
        if len(keep_dims[layer]) == 0:
            hubert.encoder.layer[layer].intermediate.dense = None
            hubert.encoder.layer[layer].output.dense = None
        else:
            hubert.encoder.layer[
                layer].intermediate.dense = prune_linear_layer(
                    hubert.encoder.layer[layer].intermediate.dense,
                    index=torch.LongTensor(keep_dims[layer]).to(device),
                    dim=0)
            hubert.encoder.layer[layer].output.dense = prune_linear_layer(
                hubert.encoder.layer[layer].output.dense,
                index=torch.LongTensor(keep_dims[layer]).to(device),
                dim=1)


def load_zs(model_path):
    if model_path.endswith("zs.pt"):
        zs_path = model_path
    else:
        zs_path = os.path.join(model_path, "zs.pt")

    if os.path.exists(zs_path):
        zs = torch.load(zs_path, map_location="cpu")
        if zs is None:
            model_path = os.path.dirname(model_path)
            l0_module = torch.load(os.path.join(model_path, "l0_module.pt"),
                                   map_location="cpu")
            zs = l0_module.forward(training=False)
        return zs
    else:
        return None


def load_pruned_model(model, weights):
    config = model.config
    dim_per_head = config.hidden_size // config.num_attention_heads
    zs = {}

    hubert_name = "w2v_model"

    hidden_z = torch.zeros(config.hidden_size)
    hidden_z[:weights[f"{hubert_name}.post_proj.weight"].shape[1]] = 1
    zs["hidden_z"] = hidden_z

    head_z = torch.zeros(config.num_hidden_layers, config.num_attention_heads)
    head_layer_z = torch.zeros(config.num_hidden_layers)
    for i in range(config.num_hidden_layers):
        key = f"{hubert_name}.encoder.layer.{i}.attention.output.dense.weight"
        if key in weights:
            remaining_heads = weights[key].shape[-1] // dim_per_head
            head_z[i, :remaining_heads] = 1
            head_layer_z[i] = 1
    zs["head_z"] = head_z
    zs["head_layer_z"] = head_layer_z

    int_z = torch.zeros(config.num_hidden_layers, config.intermediate_size)
    mlp_z = torch.zeros(config.num_hidden_layers)
    for i in range(config.num_hidden_layers):
        key = f"{hubert_name}.encoder.layer.{i}.output.dense.weight"
        if key in weights:
            remaining_int_dims = weights[key].shape[-1]
            int_z[i, :remaining_int_dims] = 1
            mlp_z[i] = 1
    zs["intermediate_z"] = int_z
    zs["mlp_z"] = mlp_z

    prune_model_with_z(zs, model)
    model.load_state_dict(weights, strict=False)
    return model


def get_full_model_size(model_class, model_name):
    model = model_class.from_pretrained(model_name)
    model_size = calculate_parameters(model)
    return model_size


def print_zs(zs_path):
    """
This is a groups style docs.
 
Parameters:
 param1 - this is the first param
 param2 - this is a second param
 
Returns:
 This is a description of what is returned
 
Raises:
 KeyError - raises an exception
"""
    zs_module = torch.load(zs_path)
    print(zs_module.keys())
    print(zs_module)


def print_model(model_path):
    """
This is a groups style docs.
 
Parameters:
 param1 - this is the first param
 param2 - this is a second param
 
Returns:
 This is a description of what is returned
 
Raises:
 KeyError - raises an exception
"""
    model = torch.load(model_path)
    print(model.keys())
    print(model)