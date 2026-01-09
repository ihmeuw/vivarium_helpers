import collections

import pandas as pd

from ..utils import _ensure_columns_not_levels, _ensure_iterable, list_columns

VALUE_COLUMN = "value"
DRAW_COLUMN = "input_draw"
SCENARIO_COLUMN = "scenario"
MEASURE_COLUMN = "measure"

INDEX_COLUMNS = [DRAW_COLUMN, SCENARIO_COLUMN]


class VPHOperator:
    """Class to perform operations on Vivarium Public Health data."""

    def __init__(
        self,
        value_col=None,
        draw_col=None,
        scenario_col=None,
        measure_col=None,
        index_cols=None,
    ):
        self.value_col = VALUE_COLUMN if value_col is None else value_col
        self.draw_col = DRAW_COLUMN if draw_col is None else draw_col
        self.scenario_col = SCENARIO_COLUMN if scenario_col is None else scenario_col
        self.measure_col = MEASURE_COLUMN if measure_col is None else measure_col
        self.index_cols = (
            [self.draw_col, self.scenario_col] if index_cols is None else index_cols
        )

    def set_index_columns(self, index_columns: list) -> None:
        """
        Set INDEX_COLUMNS to a custom list of columns for the Vivarium model output.
        For example, if tables for different locations have been concatenated with
        a new column called 'location', then use the following to get the correct
        behavior for the functions in this module:

        set_index_columns(['location']+lsff_output_processing.INDEX_COLUMNS)
        """
        # TODO: Save current index columns, and add a method .reset_index()
        # to return to the original
        self.index_cols = index_columns

    # TODO: Either add an index_cols parameter here, or figure out if
    # there's a way to mimic the same behavior by allowing the user
    # to pass both `include` and `exclude`
    def value(self, df, include=None, exclude=None, value_cols=None):
        """Set the index of the dataframe so that its only column(s) is (are) value_cols.
        This is useful for performing arithmetic on the dataframe, e.g. value(df1) + value(df2),
        assuming the resulting dataframes have compatible indices.
        - If neither `include` nor `exclude` are specified, the index of the dataframe will be
          set to all columns except those in `value_cols`.
        - If `include` is not None, the index will be set to the columns specified in `include`
          plus those specified in INDEX_COLUMNS (typically DRAW_COLUMN and SCENARIO_COLUMN).
        - If `exclude` is not None, the columns listed in `exclude` will be excluded from the index.
        - If both `include` and `exclude` are specified, a ValueError is raised - you can specify either
          columns to include or exclude, but not both.
        This function will still work if some of the non-value columns of df have been moved into a MultiIndex.
        If the index consists of a single column and the intent is to add additional non-value columns
        to the index, the caller should call df.reset_index() before passing to this function, or else
        the existing index column will be dropped and replaced with the others.
        """
        if value_cols is None:
            value_cols = self.value_col
        df = _ensure_columns_not_levels(
            df, list_columns(include, exclude, value_cols, default=[])
        )
        value_cols = _ensure_iterable(value_cols)
        if include is None:
            exclude = _ensure_iterable(exclude, default=[])
            index_cols = df.columns.difference([*value_cols, *exclude]).to_list()
        elif exclude is not None:
            raise ValueError(
                "Only one of `include` or `exclude` can be specified."
                f" You passed {include=}, {exclude=}"
            )  # syntax requires python >=3.8
        else:
            include = _ensure_iterable(include, df)
            index_cols = [*include, *self.index_cols]
        return df.set_index(index_cols)[value_cols]

    def marginalize(
        self,
        df: pd.DataFrame,
        marginalized_cols,
        value_cols=None,
        reset_index=True,
        func="sum",
        args=(),  # Positional args to pass to func in DataFrameGroupBy.agg
        **kwargs,  # Keywords to pass to DataFrameGroupBy.agg
    ) -> pd.DataFrame:
        """Sum (or otherwise aggregate) the values of a dataframe over
        the specified columns to marginalize out.

        https://en.wikipedia.org/wiki/Marginal_distribution

        The `marginalize` and `stratify` functions are complementary in that the two functions do the same thing
        (sum values of a dataframe over a subset of the dataframe's columns), but the specified columns
        in the second argument of the two functions are opposites:
            For `marginalize` you specify the marginalized columns you want to sum over, whereas
            for `stratify` you specify the stratification columns that you want to keep disaggregated.

        Parameters
        ----------

        df: DataFrame
            A dataframe with at least one "value" column to be aggregated, and additional "identifier" columns,
            at least one of which is to be marginalized out. That is, the data in the "value" column(s) will be summed
            over all catgories in the "marginalized" column(s). All columns in the dataframe are assumed to be either
            "value" columns or "identifier" columns, and the columns to marginalize should be a subset of the
            identifier columns.

        martinalized_cols: single column label, list of column labels, or pd.Index object
            The column(s) to sum over (i.e. marginalize)

        value_cols: single column label, list of column labels, or pd.Index object
            The column(s) in the dataframe that contain the values to sum

        reset_index: bool, default True
            Whether to reset the dataframe's index after calling groupby().agg()

        func : function, str, list, dict or None, default 'sum'
            The function argument to pass to groupby(...).agg()

        Returns
        ------------
        summed_data: DataFrame
            DataFrame with the aggregated values, whose columns are the same as those in df except without `marginalized_cols`,
            which have been aggregated over.
            If reset_index == False, all the resulting columns will be placed in the DataFrame's index except for `value_cols`.
        """
        if value_cols is None:
            value_cols = self.value_col
        marginalized_cols = _ensure_iterable(marginalized_cols)
        value_cols = _ensure_iterable(value_cols)
        # Move Index levels into columns to enable passing index
        # level names as well as column names to marginalize
        df = _ensure_columns_not_levels(df, marginalized_cols)
        groupby_cols = df.columns.difference(
            # must convert Index to list for groupby to work properly
            [*marginalized_cols, *value_cols]
        ).to_list()
        aggregated_data = df.groupby(
            # observed=True needed for Categorical data
            groupby_cols,
            as_index=(not reset_index),
            observed=True,
        )[value_cols].agg(func, *args, **kwargs)
        return aggregated_data

    def stratify(
        self,
        df: pd.DataFrame,
        strata,
        value_cols=None,
        index_cols=None,
        reset_index=True,
        func="sum",
        args=(),  # Positional args to pass to func in DataFrameGroupBy.agg
        **kwargs,  # Keywords to pass to DataFrameGroupBy.agg
    ) -> pd.DataFrame:
        """Sum (or otherwise aggregate) the values of the dataframe so
        that the reult is stratified by the specified strata.

        https://en.wikipedia.org/wiki/Stratification_(clinical_trials)

        More specifically, `stratify` groups `df` by the stratification columns and sums the value columns,
        but automatically adds INDEX_COLS (usually DRAW_COLUMN and SCENARIO_COLUMN) to the `by` parameter
        of the groupby. That is, the return value is df.groupby(strata+INDEX_COLS)[value_cols].sum()

        The `marginalize` and `stratify` functions are complementary in that the two functions do the same thing
        (sum values of a dataframe over a subset of the dataframe's columns), but the specified columns
        in the second argument of the two functions are opposites:
            For `marginalize` you specify the marginalized columns you want to sum over, whereas
            for `stratify` you specify the stratification columns that you want to keep disaggregated.

        Parameters
        ----------

        df: DataFrame
            A dataframe with at least one "value" column to be aggregated, and additional "identifier" columns
            which must include those listed in INDEX_COLS and potentially other columns to stratify by.
            That is, the data in the "value" column(s) will be summed over all catgories in the identifier
            column except those in `strata` and INDEX_COLS. All columns in the dataframe are assumed to be either
            "value" columns or "identifier" columns, and the columns to stratify by should be a subset of the
            identifier columns.

        strata: single column label, list of column labels, or pd.Index object
            The column(s) to stratify by (i.e. group by before summing)

        value_cols: single column label, list of column labels, or pd.Index object
            The column(s) in the dataframe that contain the values to sum

        reset_index: bool
            Whether to reset the dataframe's index after calling groupby().sum()

        Returns
        ------------

        summed_data: DataFrame
            DataFrame with the summed values, whose columns are the columns listed in `strata` and INDEX_COLS
            (usually DRAW_COLUMN and SCENARIO_COLUMN), with all other columns being marginalized out.
            If reset_index == False, all the resulting columns will be placed in the DataFrame's index except
            for `value_cols`.
        """
        if value_cols is None:
            value_cols = self.value_col
        if index_cols is None:
            index_cols = self.index_cols
        strata = _ensure_iterable(strata)
        value_cols = _ensure_iterable(value_cols)
        groupby_cols = [*strata, *index_cols]
        aggregated_data = df.groupby(groupby_cols, as_index=(not reset_index), observed=True)[
            value_cols
        ].agg(func, *args, **kwargs)
        return aggregated_data

    def aggregate_categories(
        self,
        df,
        category_col,
        supercategory_to_categories,
        append=False,
        value_cols=None,
        reset_index=True,
        func="sum",
        args=(),  # Positional args to pass to func in DataFrameGroupBy.agg
        **kwargs,  # Keywords to pass to DataFrameGroupBy.agg
    ):
        """Aggregates (via sum by default) the values corresponding to the specified categories in
        a specified column of a dataframe into "supercategories" for that column.

        The categories to aggregate and the supercategories they comprise are specified by the dictionary
        `supercategory_to_categories`, which maps each supercategory name to the list of categories that
        make up the supercategory.

        Returns a new dataframe with the same columns as the argument `df`.
        If append is False (default), the new dataframe contains only the aggregated supercategories,
        in the `category_col` column.
        If append is True, the aggregated supercategories are appended to the end of the dataframe df,
        and this new concatnated dataframe is returned.

        I have come across at least four use cases for this function in
        the CIFF acute malnutrition project:
        * Aggregating age groups into 'all_ages' or, e.g., 'over_6_months' (this could apply to any
          additive measure, e.g., person time, wasting prevalence, or DALYs).
        * Aggregating causes into 'all_causes' (e.g., for deaths or DALYs).
        * Aggrgating the SAM and MAM wasting states into a 'global_acute_malnutrition' "superstate,"
          and aggregating the TMREL and MILD wasting states into a 'no_acute_malnutrition' "superstate,"
          to match the common definitions of "wasted" and "not wasted" (this could apply to prevalence
          or to any additive measure stratified by wasting state).
        * Aggregating transition counts to calculate the total inflow into a wasting state.

        Parameters
        ----------

        df: DataFrame
            The dataframe containing the category column to aggregates.

        category_col: str
            The name of the column containing the categories to aggregate.

        supercategory_to_categories: dict
            A dictionary mapping the name of each supercategory to a list of the categories
            comprising the supercategory, or to a single category.
            Any category not appearing in one of the lists of categories
            for a supercategory will get mapped to itself.

        append: bool, default False
            Whether to append the aggregated supercategories to the original dataframe or to
            return a dataframe containing only the aggregated supercategories.'

        Returns
        -------
        aggregated_df: DataFrame
            A new dataframe which contains the aggregated supercategories, appended to the end of
            `df` if `append` is True, or containing only the supercategories if `append` is False.
        """
        # Reverse the dictionary, unpacking the lists
        category_to_supercategory = {
            category: supercategory
            for supercategory, categories in supercategory_to_categories.items()
            for category in _ensure_iterable(categories)
        }
        # Margnializing over nothing is equivalent to stratifying by
        # every column except those in value_cols. After mapping each
        # category to its supercategory, this will collapse all rows
        # corresponding to each supercategory within the remaining
        # strata.
        aggregated_df = df.replace({category_col: category_to_supercategory}).pipe(
            self.marginalize, [], value_cols, reset_index, func, args, **kwargs
        )
        if append:
            return pd.concat([df, aggregated_df], ignore_index=True)
        else:
            return aggregated_df

    def ratio(
        self,
        numerator: pd.DataFrame,
        denominator: pd.DataFrame,
        strata,
        multiplier=1,
        numerator_broadcast=None,
        denominator_broadcast=None,
        value_col=None,
        index_cols=None,
        measure_col=None,
        dropna=False,
        record_inputs=None,
        reset_index=True,
    ) -> pd.DataFrame:
        """
        Compute a ratio or rate by dividing the numerator by the denominator.

        Parameters
        ----------

        numerator : DataFrame
            The numerator data for the ratio or rate.

        denominator : DataFrame
            The denominator data for the ratio or rate.

        strata : list of column names present in the numerator and denominator (also accepts a single column name)
            The stratification variables for the ratio or rate.

        multiplier : int or float, default 1
            Multiplier for the numerator, typically a power of 10,
            to adjust the units of the result. For example, if computing a ratio,
            some multipliers with corresponding units are:
            1 - proportion
            100 - percent
            1000 - per thousand
            100_000 - per hundred thousand

        numerator_broadcast : list of column names present in the numerator, or None
            Additional columns in the numerator by which to stratify or broadcast.
            Note that the population in the numerator must always be a subset of
            the population in the denominator, so it only makes sense to include
            addiional strata in the numerator.

            For example, if 'sex' is included in `numerator_broadcast` but not `strata`,
            then the resulting ratio can be interpreted as a joint distribution over sex,
            and summing over the 'sex' column in the ratio would give the same result as
            passing numerator_broadcast=None.

            You can also pass columns to `numerator_broadcast` to do muliple computations
            at the same time. E.g. pass 'cause' to compute a ratio or rate for multiple causes
            at once, or pass 'measure' to compute a ratio or rate for multiple measures at
            once (like deaths, ylls, and ylds).

        denominator_broadcast : list of column names present in the denominator, or None
            Additional columns in the numerator by which to broadcast.

        value_col : single column name (a singleton list is also accepted), default VALUE_COLUMN
            The column where the values in the numerator and denominator dataframes are stored.

        measure_col: single column name (a singleton list is also accepted), default MEASURE_COLUMN
            The column indicating the type of measure stored in the numerator and denominator dataframes.
            Not used if `record_inputs` is False.

        dropna : boolean, default False
             Whether to drop rows with NaN values in the result, namely
             if division by 0 occurs because of an empty stratum in the denominator.

        record_inputs : boolean or None, default None
            Whether to record the multiplier and the numeraor's and denominator's measures in the output.
            If None, defaults to the value of `reset_index` to facilitate performing further operations with
            the ratio if reset_index == False.

        reset_index : boolean, default True
            Whether to move index levels back into the dataframe's columns after computing the ratio.
            If reset_index==False and record_inputs==False, the only column of the returned dataframe
            will be `value_col`, which can facilitate performing further operations on the ratio (e.g.
            multiplying by a constant or combining with another dataframe that has the same index).

         Returns
         -------
         ratio : DataFrame
             The ratio or rate data = numerator / denominator.
        """
        if value_col is None:
            value_col = self.value_col
        if measure_col is None:
            measure_col = self.measure_col
        # Ensure that index columns in numerator and denominator are columns not index levels,
        # to guarantee that _ensure_iterable will work and df[measure_col] will work.
        numerator = _ensure_columns_not_levels(numerator)
        denominator = _ensure_columns_not_levels(denominator)
        # Ensure that numerator_broadcast and denominator_broadcast are iterables of column names
        numerator_broadcast = _ensure_iterable(numerator_broadcast, default=[])
        denominator_broadcast = _ensure_iterable(denominator_broadcast, default=[])

        # Avoid potential confusion by requiring common stratification columns to go in strata.
        if len(set(numerator_broadcast) & set(denominator_broadcast)) > 0:
            raise ValueError(
                "`numerator_broadcast` and `denominator_broadcast` must be disjoint lists of column names."
                " Any column to include in both the numerator and denominator should go in `strata`."
            )

        # Default behavior is to record inputs only if index is reset
        if record_inputs is None:
            record_inputs = reset_index

        if record_inputs:
            # Really I think the 'measure' column should always have a
            # unique value, but currently that is not the case for
            # transition counts...
            numerator_measure = "|".join(numerator[measure_col].unique())
            denominator_measure = "|".join(denominator[measure_col].unique())

        # Ensure strata is an iterable of column names so it can be
        # concatenated with broadcast columns
        strata = _ensure_iterable(strata)
        # Stratify numerator and denominator with broadcast columns included
        numerator = self.stratify(
            numerator,
            [*strata, *numerator_broadcast],
            value_cols=value_col,
            index_cols=index_cols,
            reset_index=False,
        )
        denominator = self.stratify(
            denominator,
            [*strata, *denominator_broadcast],
            value_cols=value_col,
            index_cols=index_cols,
            reset_index=False,
        )

        # Compute the ratio
        ratio = (numerator / denominator) * multiplier

        # If dropna is True, drop rows where we divided by 0
        if dropna:
            # Note: inplace=True is deprecated because .dropna() changes
            # the shape of data and never actually operates in place
            ratio = ratio.dropna()

        if record_inputs:
            ratio[f"numerator_{measure_col}"] = numerator_measure
            ratio[f"denominator_{measure_col}"] = denominator_measure
            ratio["multiplier"] = multiplier

        if reset_index:
            # Note: inplace=True is deprecated for .reset_index because
            # it becomes redundant when Copy-on-Write is implemented:
            # https://github.com/pandas-dev/pandas/blob/2654fe9ba91dd44ed66f0c246d4ed0c5dde7e391/web/pandas/pdeps/0008-inplace-methods-in-pandas.md
            ratio = ratio.reset_index()

        return ratio

    # TODO: Generalize the difference function to linear_combination,
    # e.g. to take a weighted average of a 0% coverage scenario
    # with a 100% coverage scenario
    def difference(
        self,
        measure: pd.DataFrame,
        identifier_col: str,
        minuend_id=None,
        subtrahend_id=None,
    ) -> pd.DataFrame:
        """
        Returns the difference of a measure stored in the measure DataFrame, where the
        rows for the minuend (that which is diminished) and subtrahend (that which is subtracted)
        are determined by the values in identifier_col
        """
        if minuend_id is not None:
            minuend = measure[measure[identifier_col] == minuend_id]
            if subtrahend_id is not None:
                subtrahend = measure[measure[identifier_col] == subtrahend_id]
            else:
                # Use all values not equal to minuend_id for subtrahend (minuend will be broadcast over subtrahend)
                subtrahend = measure[measure[identifier_col] != minuend_id]
        elif subtrahend_id is not None:
            subtrahend = measure[measure[identifier_col] == subtrahend_id]
            # Use all values not equal to subtrahend_id for minuend (subtrahend will be broadcast over minuend)
            minuend = measure[measure[identifier_col] != subtrahend_id]
        else:
            raise ValueError(
                "At least one of `minuend_id` and `subtrahend_id` must be specified"
            )

        # Columns to match when subtracting subtrahend from minuend
        # Oh, I just noticed that I could use the Index.difference() method here, which I was unaware of before...
        index_columns = sorted(
            set(measure.columns) - set([identifier_col, VALUE_COLUMN]),
            key=measure.columns.get_loc,
        )

        minuend = minuend.set_index(index_columns)
        subtrahend = subtrahend.set_index(index_columns)

        # Add the identifier column to the index of the larger dataframe
        # (or default to the subtrahend dataframe if neither needs broadcasting).
        if minuend_id is None:
            minuend.set_index(identifier_col, append=True, inplace=True)
        else:
            subtrahend.set_index(identifier_col, append=True, inplace=True)

        # Subtract DataFrames, not Series, because Series will drop the identifier column from the index
        # if there is no broadcasting. (Behavior for Series and DataFrames is different - is this a
        # feature or a bug in pandas?)
        difference = minuend[[self.value_col]] - subtrahend[[self.value_col]]
        difference = difference.reset_index()

        # Add a column to specify what was subtracted from (the minuend) or what was subtracted (the subtrahend)
        colname, value = (
            ("subtracted_from", minuend_id)
            if minuend_id is not None
            else ("subtracted_value", subtrahend_id)
        )
        difference.insert(difference.columns.get_loc(identifier_col) + 1, colname, value)

        return difference

    def averted(self, measure: pd.DataFrame, baseline_scenario: str, scenario_col=None):
        """
        Compute an "averted" measure (e.g. DALYs) or measures by subtracting
        the intervention value from the baseline value.

        Parameters
        ----------

        measure : DataFrame
            DataFrame containing both the baseline and intervention data.

        baseline_scenario : scalar, typically str
            The name or other identifier for the baseline scenario in the
            `scenario_col` column of the `measure` DataFrame.

        scenario_col : str, default None
            The name of the scenario column in the `measure` DataFrame.
            Defaults to the global parameter SCENARIO_COLUMN if None is passed.

        Returns
        -------

        averted : DataFrame
            The averted measure(s) = baseline - intervention
        """
        if scenario_col is None:
            scenario_col = self.scenario_col
        # Subtract intervention from baseline
        averted = self.difference(
            measure, identifier_col=scenario_col, minuend_id=baseline_scenario
        )
        # Insert a column after the scenario column to record what the baseline scenario was
        #     averted.insert(averted.columns.get_loc(scenario_col)+1, 'relative_to', baseline_scenario)
        return averted

    def describe(self, df, **describe_kwargs):
        """Describes the distribution of df's values across draws.
        This is a wrapper function for DataFrameGroupBy.describe(),
        with `df` grouped by everything except draw and value.
        """
        if "percentiles" not in describe_kwargs:
            describe_kwargs["percentiles"] = [0.025, 0.975]
        excluded_cols = [self.draw_col, self.value_col]
        df = _ensure_columns_not_levels(df, excluded_cols)
        groupby_cols = df.columns.difference(excluded_cols).to_list()
        return df.groupby(groupby_cols)[self.value_col].describe(**describe_kwargs)

    def assert_values_equal(self, df1, df2, **kwargs):
        """Test whether the value columns of df1 and df2 are equal, using all other columns as the index,
        using the `pd.testing.assert_frame_equal` function.
        """
        df1 = self.value(df1)
        df2 = self.value(df2).reindex(df1.index)
        pd.testing.assert_frame_equal(df1, df2, **kwargs)

    def compare_values(self, df1, df2, **kwargs):
        """Compare the value columns of df1 and df2, using all other columns as the index,
        using the `pd.DataFrame.compare` method.
        """
        df1 = self.value(df1)
        df2 = self.value(df2).reindex(df1.index)
        return df1.compare(df2, **kwargs)


# NOTE:
# An alternative strategy to defining __getattr__ as below
# is to replace this module in sys.modules with the global instance
# of the class defined above (called _ops below), as described here:
# https://stackoverflow.com/questions/2447353/getattr-on-a-module
#
# Specifically, in this answer and the page it links to:
# https://stackoverflow.com/a/7668273/24446049
# https://mail.python.org/pipermail/python-ideas/2012-May/014969.html

# Global instance to implement methods as module functions,
# using default values for columns names
_ops = VPHOperator()

# Enable calling all methods of VPHOperator as module functions
def __getattr__(name):
    try:
        return getattr(_ops, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
