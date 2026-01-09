import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype


def sample_array_from_propensity(
    propensity,
    categories,
    category_cdf,
    method="select",
    default_category=None,
):
    """Sample categories using the propensities.
    `propensity` is an array of numbers between 0 and 1.
    `categories` is a list/1d-array-like of categories.
    If method='select', `category_cdf` must be a mapping of categories
    to cumulative probabilities.
    If method='array', `category_cdf` must be an nd-array of cumulative
    probabilities, broadcastable to shape
    (propensity.shape, len(categories)).
    # TODO: check that this actually works when category_cdf has
    # ndim != 2, and verify that I am specifying the correct shapes
    here.
    """
    if method == "select":
        # logger.debug(f"{categories=}")
        # In numpy 2.0, it is necessary to have the default be
        # of the same dtype as the other choices
        if default_category is None:
            if isinstance(categories, pd.Series):
                default_category = categories.array[-1]
            else:
                default_category = categories[-1]
        # If category_cdf is 1-dimensional, broadcast instead of failing
        # when there is more than one propensity
        # Note: If there is only 1 propensity, this makes it so its
        # index does NOT need to be aligned with the index of a single
        # row CDF, which may or may not be what's desired.
        if isinstance(category_cdf, pd.DataFrame):
            category_cdf = category_cdf.squeeze()
        condlist = [propensity <= category_cdf[cat] for cat in categories]
        category = np.select(condlist, choicelist=categories, default=default_category)
    elif method == "array":
        if default_category is not None:
            raise ValueError("`default_category` is unsupported with method='array'")
        category_index = (
            # TODO: Explain why this works...
            np.asarray(propensity).reshape((-1, 1))
            > np.asarray(category_cdf)
        ).sum(axis=1)
        category = np.asarray(categories)[category_index]
    else:
        raise ValueError(
            f"Unknown method: {method}. " "Acceptable values are 'select' and 'array'."
        )
    return category


def sample_categorical_from_propensity(
    propensity,
    # pandas CategoricalDtype or iterable of unique categories
    categories,
    category_cdf,
    method="select",
    default_category=None,
    ordered=None,
):
    # If we know the CategoricalDtype from categories,
    # set the `categories` and `dtype` parameters
    # for passing into Categorical.from_codes
    if isinstance(categories, CategoricalDtype):
        cats = categories.categories
        dtype = categories
        categories = None
    else:
        cats = categories
        dtype = None
    codes = range(len(cats))

    if method == "select":
        # Need to change keys in category_cdf map from categories
        # to codes before passing into sample_from_array
        if isinstance(category_cdf, pd.DataFrame):
            cat_to_code = {cat: code for cat, code in zip(cats, codes)}
            category_cdf = category_cdf.rename(columns=cat_to_code)
        else:
            category_cdf = {code: category_cdf[cat] for code, cat in zip(codes, cats)}

    # Ensure that default category is valid for the CategoricalDtype
    if default_category is not None:
        if pd.isna(default_category):
            # code -1 corresponds to NaN in pandas Categoricals
            default_category = -1
        elif default_category in cats:
            # Find the code for this category
            default_category = list(cats).index(default_category)
        else:
            raise ValueError(
                "`default_category` must either be an element of `categories` "
                "or an object for which `pandas.isna` returns True."
            )

    sampled_codes = sample_array_from_propensity(
        propensity,
        codes,
        category_cdf,
        method=method,
        # Pass a default of -1 to be converted to NaN duing .from_codes,
        # which indicates that the propensity does not correspond to any
        # of the specified categories
        default_category=default_category,
    )
    sampled_categories = pd.Categorical.from_codes(
        sampled_codes, categories=categories, ordered=ordered, dtype=dtype
    )
    return sampled_categories


def sample_series_from_propensity(
    propensity,
    categories,
    category_cdf,
    method="select",
    default_category=None,
    ordered=None,
    index=None,
    dtype=None,
    name=None,
):
    is_categorical = ordered == True or isinstance(categories, CategoricalDtype)
    if is_categorical:
        if dtype is not None:
            raise ValueError(
                "`dtype` not allowed for categorical data. "
                "Pass an instance of `CategoricalDtype` to `categories` instead."
            )
        sample_array = sample_categorical_from_propensity(
            propensity,
            categories,
            category_cdf,
            method=method,
            default_category=default_category,
            ordered=ordered,
        )
    else:
        sample_array = sample_array_from_propensity(
            propensity,
            categories,
            category_cdf,
            method=method,
            default_category=default_category,
        )
    if index is None and isinstance(propensity, (pd.Series, pd.DataFrame)):
        index = propensity.index

    # Look for a name we can use for the returned Series
    if name is None:
        if isinstance(category_cdf, pd.DataFrame):
            name = category_cdf.columns.name
        if name is None and isinstance(category_cdf, pd.Series):
            name = category_cdf.index.name
        if name is None and isinstance(categories, (pd.Index, pd.Series)):
            name = categories.name
    sampled_categories = pd.Series(
        sample_array,
        index=index,
        dtype=dtype,
        name=name,
    )
    return sampled_categories
