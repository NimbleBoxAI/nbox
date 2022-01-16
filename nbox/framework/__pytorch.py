# this file has the utilities and functions required for processing pytorch items
# such as conversion to ONNX, getting the metadata and so on.

import torch


def export_to_onnx(
    model,
    args,
    export_model_path,
    input_names,
    dynamic_axes,
    output_names,
    export_params=True,
    verbose=False,
    opset_version=12,
    do_constant_folding=True,
    use_external_data_format=False,
    **kwargs
):
    torch.onnx.export(
        model,
        args=args,
        f=export_model_path,
        input_names=input_names,
        verbose=verbose,
        output_names=output_names,
        use_external_data_format=use_external_data_format,  # RuntimeError: Exporting model exceed maximum protobuf size of 2GB
        export_params=export_params,  # store the trained parameter weights inside the model file
        opset_version=opset_version,  # the ONNX version to export the model to
        do_constant_folding=do_constant_folding,  # whether to execute constant folding for optimization
        dynamic_axes=dynamic_axes,
    )


def export_to_torchscript(model, args, export_model_path, **kwargs):
    traced_model = torch.jit.trace(model, args, check_tolerance=0.0001)
    torch.jit.save(traced_model, export_model_path)


__all__ = ["export_to_onnx", "export_to_torchscript"]
