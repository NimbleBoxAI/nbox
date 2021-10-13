import nbox.framework.pytorch
import nbox.framework.sklearn

# this function is for getting the meta data and is framework agnostic, so adding this in the
# __init__ of framework submodule


def get_meta(
    input_names,
    input_shapes,
    args,
    output_names,
    output_shapes,
    outputs,
):
    # get the meta object
    def __get_struct(names_, shapes_, tensors_):
        return {
            name: {
                "dtype": str(tensor.dtype),
                "tensorShape": {"dim": [{"name": "", "size": x} for x in shapes], "unknownRank": False},
                "name": name,
            }
            for name, shapes, tensor in zip(names_, shapes_, tensors_)
        }

    meta = {"inputs": __get_struct(input_names, input_shapes, args), "outputs": __get_struct(output_names, output_shapes, outputs)}

    return meta
