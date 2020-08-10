import json

import numpy as np


def flatten_json(x, sep="__"):
    x_out = x.copy()

    while True:
        done = True

        for k, v in x.items():

            if type(v) is dict:
                done = False
                del x_out[k]

                for k_inner, v_inner in v.items():
                    x_out["{}{}{}".format(k, sep, k_inner)] = v_inner

            elif type(v) is list:
                done = False
                del x_out[k]

                for k_inner, v_inner in enumerate(v):
                    x_out["{}{}{}".format(k, sep, k_inner)] = v_inner

        if done:
            break

        x = x_out.copy()

    return x_out


class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):

            return int(obj)

        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        if isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}

        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        if isinstance(obj, (np.bool_)):
            return bool(obj)

        if isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)
