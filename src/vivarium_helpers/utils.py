import collections
import re

import pandas as pd


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


class AttributeMapping(FrozenAttributeMapping, collections.abc.MutableMapping):
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


def _ensure_iterable(colnames, default=None):
    """Wrap a single column name in a list, or return colnames unaltered if it's already a list of column names.
    If colnames is None, its value will first be set to the default value (e.g. pass `default=[]` to default to
    an empty list when colnames is None).
    """

    def method1(colnames):
        """Method 1 (doesn't depend on df): Assume that if colnames has a type that is in a whitelist of
        allowed iterable types, then it is an iterable of column names, and otherwise it must be a single
        column name.
        """
        if not isinstance(colnames, (list, pd.Index)):
            colnames = [colnames]
        return colnames

    if colnames is None:
        colnames = default
    return method1(colnames)  # Go with the most restrictive method for now


def _ensure_columns_not_levels(df, column_list=None):
    """Move Index levels into columns to enable passing index level names as well as column names."""
    if column_list is None:
        column_list = []
    if df.index.nlevels > 1 or df.index.name in column_list:
        df = df.reset_index()
    return df


def list_columns(*column_groups, default=None) -> list:
    """Retuns a single list of column names from an arbitrary number
    of lists of column names or single column names.

    For example, all of the following return ['a', 'b', 'c', 'd', 'e']:

    list_columns(['a', 'b', 'c', 'd', 'e'])
    list_columns('a', 'b', 'c', 'd', 'e')
    list_columns(['a', 'b'], ['c', 'd'], ['e'])
    list_columns('a', ['b', 'c'], 'd', ['e'])
    ...
    etc.
    """
    return [
        col
        for col_or_cols in column_groups
        for col in _ensure_iterable(col_or_cols, default=default)
    ]


def convert_to_variable_name(string):
    """Converts a string to a valid Python variable.
    Runs of non-word characters (regex matchs \W+) are converted to '_',
    and '_' is appended to the beginning of the string if the string
    starts with a digit (regex matches ^(?=\d)).

    Solution copied from here:
    https://stackoverflow.com/questions/3303312/how-do-i-convert-a-string-to-a-valid-variable-name-in-python
    """
    return re.sub("\W+|^(?=\d)", "_", string)


def column_to_ordered_categorical(df, colname, ordered_categories, inplace=False):
    """Converts the column `colname` of the DataFrame `df` to an orderd
    pandas Categorical. This is useful for automatically displaying
    unique column elements in a specified order in results tables or
    plots.
    """
    categorical = pd.Categorical(df[colname], categories=ordered_categories, ordered=True)
    if inplace:
        df[colname] = categorical
        return None
    else:
        return df.assign(**{colname: categorical})


def get_mean_lower_upper(
    described_data, colname_mapper={"mean": "mean", "2.5%": "lower", "97.5%": "upper"}
):
    """
    Gets the mean, lower, and upper value from `described_data` DataFrame, which is assumed to have
    the format resulting from a call to DataFrame.describe().
    """
    return described_data[colname_mapper.keys()].rename(columns=colname_mapper).reset_index()


# Alternative strategy to the get_mean_lower_upper function above
def aggregate_mean_lower_upper(df_or_groupby, lower_rank=0.025, upper_rank=0.975):
    """Get mean, lower, and upper from a DataFrame or GroupBy object."""

    def lower(x):
        return x.quantile(lower_rank)

    def upper(x):
        return x.quantile(upper_rank)

    return df_or_groupby.agg(["mean", lower, upper])


def aggregate_with_join(strings, sep="|"):
    """Combines an iterable of strings into a single string by calling
    sep.join(strings).
    """
    return sep.join(strings)
