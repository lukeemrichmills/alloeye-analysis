import numpy as np

from src.d01_data.database.Errors import InvalidValue


class FeatureWrapper:
    invalid_dtype_num = 101010
    invalid_dypt_str = "101010"
    null_float = np.nan
    null_int = 909090
    null_str = "NULL"
    num_too_low = -999
    num_too_high = 999
    str_too_short = "TOO_SHORT"
    str_too_long = "TOO_LONG"

    def __init__(self, function, input_args, out_dtype, bounds_args, null_value="default"):
        self.out_dtype = out_dtype
        self.null_value = self.get_default_null(out_dtype) if null_value == "default" else null_value
        self.out = self.check_output(function, input_args, bounds_args)

    def check_output(self, function, input_args, bounds_kwargs):
        out = function(*input_args)
        if self.dtype_valid(out):
            return self.check_bounds(out, bounds_kwargs)
        else:
            raise InvalidValue(out, "float, int or str",
                               message=f"Invalid output data type {type(out)} for feature output {out}")

    def get_default_null(self, out_dtype):
        if out_dtype is float:
            return FeatureWrapper.null_float
        elif out_dtype is int:
            return FeatureWrapper.null_int
        elif out_dtype is str:
            return FeatureWrapper.null_str
        else:
            raise InvalidValue(out_dtype, "float, int or str",
                               message="Invalid feature data type when getting null value")

    def dtype_valid(self, out):
        if self.out_dtype is float:
            dtype_check = type(out) is float or np.issubdtype(type(out), np.floating)
        elif self.out_dtype is int:
            dtype_check = (type(out) is int or np.issubdtype(type(out), np.integer)
                           or type(out) is float or np.issubdtype(type(out), np.floating))
        elif self.out_dtype is str:
            dtype_check = type(out) is str
        return dtype_check or out == self.null_value or out is None

    def check_bounds(self, out, bounds_args):
        if out == self.null_value:
            return out
        if self.out_dtype is float or self.out_dtype is int:
            return self.check_num_bounds(out, *bounds_args)
        elif self.out_dtype is str:
            return self.check_str_bounds(out, *bounds_args)

    def check_num_bounds(self, out, lower, upper):
        if out is None:
            return out
        elif lower <= out <= upper:
            return out
        elif out < lower:
            return FeatureWrapper.num_too_low
        elif out > upper:
            return FeatureWrapper.num_too_high
        else:
            return out

    def check_str_bounds(self, out, too_short, too_long):
        if out is None:
            return out
        elif too_short <= len(out) <= too_long:
            return out
        elif len(out) < too_short:
            return FeatureWrapper.str_too_short
        elif len(out) > too_long:
            return FeatureWrapper.str_too_long
        else:
            return out
