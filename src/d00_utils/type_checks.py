from src.d01_data.database.Errors import InvalidValue


def type_or_list(type_in, input_in):
    if isinstance(input_in, type_in):
        return [input_in]
    elif not isinstance(input_in, list) or not all(isinstance(elem, type_in) for elem in input_in):
        message = "invalid input"
        raise InvalidValue(True, False, message)
    else:
        return input_in
