
def del_multiple(index, l):
    for _list in l:
        del _list[index]
    return l


def input_to_continue(message):
    input_message = f'{message}'
    in_data = input(input_message)
    return in_data


def shift_list_down(input_list, blank_ph, keep_length=True):
    """ if keep_length True then will cut off last item from list"""
    output_list = [blank_ph]
    index = 1 if keep_length else 0
    output_list.extend(input_list[:-index])
    return output_list


def shift_list_up(input_list, blank_ph, keep_length=True):
    index = 1 if keep_length else 0
    output_list = input_list[index:]
    output_list.append(blank_ph)
    return output_list
