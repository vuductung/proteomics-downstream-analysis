import os
import ipykernel
from IPython.core.getipython import get_ipython
import textwrap

def is_jupyter_notebook():
    return get_ipython() is not None and isinstance(get_ipython(), ipykernel.zmqshell.ZMQInteractiveShell)

def format_ytick_label(ytick):
    max_width = 15
    words = ytick.split()
    res = ''
    line = ''
    for w in words:
        line += (w + ' ')
        if len(line) > max_width:
            res += line.strip() + '<br>'
            line = ''
            
    res += line.strip()
    return res

def float_string_split(data):
    return data.select_dtypes(float), data.select_dtypes('string')

def intersection_of_list_of_lists(list_of_lists):
    if not list_of_lists:
        return []  # Return an empty list if the input is empty

    # Initialize the intersection set with the first list
    intersection_set = set(list_of_lists[0])
    
    # Iterate over the remaining lists and update the intersection set
    for lst in list_of_lists[1:]:
        intersection_set.intersection_update(lst)
    
    # Convert the intersection set back to a list (optional, depending on the requirement)
    return list(intersection_set)

def wrap_labels(text, width):
    return textwrap.fill(text, width)

def add_suffixes(original_list):
    result = []
    counts = {}
    
    for item in original_list:
        if item in counts:
            counts[item] += 1
            result.append(f"{item}_{counts[item]}")
        else:
            counts[item] = 0
            result.append(item)
    
    return result

def filter_dict_by_key_content(dictionary, substring):
    """
    Filter a dictionary to keep only key-value pairs where the key contains a specific substring.
    
    Args:
    dictionary (dict): The input dictionary to filter.
    substring (str): The substring to search for in the keys.
    
    Returns:
    dict: A new dictionary containing only the key-value pairs where the key contains the substring.
    """
    return {key: value for key, value in dictionary.items() if substring in key}


def add_suffixes_to_duplicates(df, column):

    """add suffixes to columns with duplicate values"""
    # Overwrite the column with suffixed values for duplicates
    df[column] = (
        df.groupby(column)[column]
        .transform(lambda x: x + '_' + (x.groupby(x).cumcount() + 1).astype(str))
        .mask(lambda x: ~x.duplicated(), df[column])
    )
    return df
