import collections
import re

class FrozenAttributeMapping(collections.abc.Mapping):
    """Implementation of the Mapping abstract base class that
    stores mapping keys as object attributes. This is a convenience
    whose main purpose is to allow tab completion of mapping keys
    in an interactive coding environment.
    """
    def __init__(self, mapping=(), /, **kwargs):
        """Create a FrozenAttributeMapping object from a dictionary
        or other implementation of Mapping that maps keys to values.
        Dictionary keys become object attributes that store the
        corresponding dictionary values.

        The constructor works the same way as for dictionaries:

        class FrozenAttributeMapping(**kwargs)
        class FrozenAttributeMapping(mapping, **kwargs)
        class FrozenAttributeMapping(iterable, **kwargs)

        See the Python documentation for more details:
        https://docs.python.org/3/library/stdtypes.html#dict

        The keys in `mapping` must be valid Python variable names
        (this is not enforced in the constructor, but bad variable
        names will cause problems when trying to access the values
        via attributes).
        """
        # Passing () by default for mapping results in the same
        # behavior as dictionaries for any argument combination
        self.__dict__ = dict(mapping, **kwargs)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def to_dict(self):
        return dict(self.__dict__)

    def vars(self):
        # Does the same thing as `to_dict`
        return vars(self)

    # Overrides default implementation in abc.Mapping
    def keys(self):
        return self.__dict__.keys()

class AttributeMapping(
    FrozenAttributeMapping, collections.abc.MutableMapping):
    """The constructor works the same way as for dictionaries:

    class AttributeMapping(**kwargs)
    class AttributeMapping(mapping, **kwargs)
    class AttributeMapping(iterable, **kwargs)

    See the Python documentation for more details:
    https://docs.python.org/3/library/stdtypes.html#dict
    """

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, key):
        del self.__dict__[key]

def convert_to_variable_name(string):
    """Converts a string to a valid Python variable.
    Runs of non-word characters (regex matchs \W+) are converted to '_',
    and '_' is appended to the beginning of the string if the string
    starts with a digit (regex matches ^(?=\d)).
    Solution copied from here:
    https://stackoverflow.com/questions/3303312/how-do-i-convert-a-string-to-a-valid-variable-name-in-python
    """
    return re.sub('\W+|^(?=\d)', '_', string)
