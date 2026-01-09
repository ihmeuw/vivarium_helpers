import numpy as np
import pandas as pd

from .cleaning import clean_vph_output
from .loading import VPHOutput
from .operations import VPHOperator, list_columns


class VPHResults(VPHOutput):
    def __init__(
        self,
        mapping=(),
        /,
        value_col=None,
        draw_col=None,
        scenario_col=None,
        measure_col=None,
        index_cols=None,
        # record_dalys=True,
        **kwargs,
    ):
        super().__init__(mapping, **kwargs)
        self.ops = VPHOperator(value_col, draw_col, scenario_col, measure_col, index_cols)
        self._clean_vph_output()
        # if record_dalys:
        #     self.compute_and_save_dalys()

    def _clean_vph_output(self):
        """Reformat transformed count data to make more sense."""
        clean_vph_output(self)

    def table_names(self):
        return [name for name in self if name != "ops"]

    def compute_dalys(self):
        # TODO: Handle the case where one of YLLs or YLDs
        # has 'all_causes' but the other doesn't, as in NO project
        # TODO: Use .intersection and .stratify instead of
        # .difference and .marginalize
        # TODO: Perhaps add a function to concatenate all 4 burden dataframes,
        # also by taking the intersection of all stratification variables
        yll_extra_strata = self.ylls.columns.difference(self.ylds.columns)
        yld_extra_strata = self.ylds.columns.difference(self.ylls.columns)
        # Marginalize extra columns so that we can concatenate
        # print(f'{yll_extra_strata=}, {yld_extra_strata=}')
        ylls = self.ops.marginalize(self.ylls, yll_extra_strata)
        ylds = self.ops.marginalize(self.ylds, yld_extra_strata)
        dalys = pd.concat([ylls, ylds])
        # print(f'{len(ylls)=}, {len(ylds)=} {len(dalys)=}')
        dalys = self.ops.aggregate_categories(dalys, "measure", {"dalys": ["ylls", "ylds"]})
        # print(f'{len(dalys)=}')
        return dalys

    def get_burden(self, measures=None):
        """Concatenate, YLDs, YLLs, DALYs, and deaths into one
        dataframe, stratified by the intersection of the stratification
        columns in these.
        """
        if measures is None:
            measures = ["deaths", "ylls", "ylds", "dalys"]
        else:
            measures = list_columns(measures)
        table_names = [measure for measure in measures if measure in self]
        # Use YLLs and YLDs to compute DALYs if we haven't already
        if "dalys" in measures and "dalys" not in table_names:
            num_missing = 0
            for measure in ["ylls", "ylds"]:
                if measure in table_names:
                    continue
                elif measure in self:
                    table_names.append(measure)
                else:
                    num_missing += 1
            if num_missing == 2:
                raise ValueError(
                    "Cannot compute DALYs without one of YLLs, YLDs, or DALYs tables"
                )
        if not table_names:  # table_names is a list, falsey if empty
            raise ValueError(f"Insufficent data tables to compute measures {measures}")
        # print(table_names)
        # Get intersection of all columns for stratification
        # (this is necessary for Nutrition Optimization project because
        #  ylds have more strata than ylls)
        columns_in_common = self[table_names[0]].columns
        # print(columns_in_common)
        for table_name in table_names[1:]:
            columns_in_common = self[table_name].columns.intersection(columns_in_common)
            # print(columns_in_common)
        strata = columns_in_common.difference([self.ops.value_col, *self.ops.index_cols])
        # print(columns_in_common)
        burdens = [self.ops.stratify(self[table_name], strata) for table_name in table_names]
        burden = pd.concat(burdens, ignore_index=True)
        # print(burden.columns)
        if "dalys" in measures and "dalys" not in table_names:
            ylls = burden.query("measure == 'ylls'")
            ylds = burden.query("measure == 'ylds'")
            # If we have comorbidity-adjusted all-cause YLDs, ensure
            # that we also have all-cause YLLs so all-cause DALYs will
            # be correct
            if "all_causes" in np.setdiff1d(ylds["cause"], ylls["cause"]):
                all_cause_ylls = self.ops.marginalize(ylls.assign(cause="all_causes"), [])
                burden = pd.concat([burden, all_cause_ylls])
                # print('yll causes:', ylls.cause.unique())
            burden = self.ops.aggregate_categories(
                burden, "measure", {"dalys": ["ylls", "ylds"]}, append=True
            )
        burden = burden.query(f"measure in {measures}").reset_index(drop=True)
        return burden

    def compute_and_save_dalys(self):
        self["dalys"] = self.get_burden("dalys")

    def compute_and_save_burden(self, measures=None):
        self["burden"] = self.get_burden(measures)

    def find_person_time_tables(self, colnames, exclude=None):
        """Generate person-time table names in this VPHResults object
        that contain the specified column names, excluding the
        specified table names.
        """
        colnames = set(list_columns(colnames))
        exclude = list_columns(exclude, default=[])
        # Create a generator of table names
        table_names = (
            table_name
            for table_name, table in self.items()
            if table_name not in exclude
            and table_name.endswith("person_time")
            and colnames.issubset(table.columns)
        )
        return table_names

    def get_person_time_table_name(self, colnames, exclude=None, error=False):
        """Return the name of a person-table that contains the
        specified columns. If no such table can be found, returns
        None unless `error` is True, in which case a ValueError
        is raised.
        """
        person_time_table_name = next(
            # Set default to None if no tables found
            self.find_person_time_tables(colnames, exclude),
            None,
        )
        if error and person_time_table_name is None:
            raise ValueError(
                f"No person-time table found with columns {colnames}."
                f" (Excluded tables: {exclude})"
            )
        return person_time_table_name

    def get_burden_rate(
        self,
        measure,
        strata,
        prefilter_query=None,
        excluded_person_time_tables=None,
        **kwargs,
    ):
        """Compute the burden rate, where burden is one of `deaths`,
        `ylls`, `ylds`, or `dalys`.

        Parameters
        ----------
        measure: str
            The measure of burden for which to compute the rate. One of
            `deaths`, `ylls`, `ylds`, or `dalys`.
        """
        # TODO: Is it possible/desirable to enable passing multiple burden
        # measures in order to compute multiple rates simultaneously?
        # Maybe it would be better to create a 'burden' table with all 4
        # measures as described above, then broadcast over measure.
        # if excluded_person_time_tables is notNone:
        #     excluded_person_time_tables = []
        burden = self[measure]
        denominator_columns = list_columns(
            strata, kwargs.get("denominator_broadcast"), default=[]
        )
        denominator_table_name = self.get_person_time_table_name(
            denominator_columns, exclude=excluded_person_time_tables, error=True
        )
        person_time = self[denominator_table_name]
        print(measure, denominator_table_name)
        # Filter input dataframes if requested
        if prefilter_query:
            burden = burden.query(prefilter_query)
            person_time = person_time.query(prefilter_query)
        # Divide to compute burden rate
        burden_rate = self.ops.ratio(
            numerator=burden,
            denominator=person_time,
            strata=strata,
            **kwargs,
            # Remove 's' from the end of measure when naming the rate
        ).assign(
            measure=f"{measure.removesuffix('s')}_rate",
            prefilter=prefilter_query,
        )
        return burden_rate

    def get_prevalence(data, state_variable, strata, prefilter_query=None, **kwargs):
        """Compute the prevalence of the specified state_variable, which may represent a risk state or cause state
        (one of 'wasting_state', 'stunting_state', or 'cause_state'), or another stratification variable
        tracked in the simulation (e.g. 'sq_lns', 'wasting_treatment', or 'x_factor').
        `prefilter_query` is a query string passed to the DataFrame.query() function of both the
        numerator and denominator before taking the ratio. This is useful for aggregating over strata
        when computing the prevalence of a subset of the population.
        The `kwargs` dictionary stores keyword arguments to pass to the vivarium_output_processing.ratio()
        function.
        """
        # Broadcast the numerator over the state variable to compute the prevalence of each state
        kwargs["numerator_broadcast"] = vop.list_columns(
            state_variable, kwargs.get("numerator_broadcast"), default=[]
        )
        # Determine columns we need for numerator and denominator so we can look up appropriate person-time tables
        numerator_columns = vop.list_columns(strata, kwargs["numerator_broadcast"])
        denominator_columns = vop.list_columns(
            strata, kwargs.get("denominator_broadcast"), default=[]
        )
        # Define numerator
        if f"{state_variable}_person_time" in data:
            state_person_time = data[f"{state_variable}_person_time"]
        else:
            # Find a person-time table that contains necessary columns for numerator.
            # Exclude cause-state person-time because it contains total person-time multiple times,
            # which would make us over-count.
            numerator_table_name = get_person_time_table_name(
                data, numerator_columns, exclude="cause_state_person_time"
            )
            state_person_time = data[numerator_table_name]
        # Find a person-time table that contains necessary columns for total person-time in the denominator.
        # Exclude cause-state person-time because it contains total person-time multiple times,
        # which would make us over-count.
        denominator_table_name = get_person_time_table_name(
            data, denominator_columns, exclude="cause_state_person_time"
        )
        person_time = data[denominator_table_name]
        # Filter input dataframes if requested
        if prefilter_query is not None:
            state_person_time = state_person_time.query(prefilter_query)
            person_time = person_time.query(prefilter_query)
        # Divide to compute prevalence
        prevalence = vop.ratio(
            numerator=state_person_time,
            denominator=person_time,
            strata=strata,
            **kwargs,  # Includes numerator_broadcast over state_variable
        ).assign(measure="prevalence")
        return prevalence

    def get_transition_rates(data, entity, strata, prefilter_query=None, **kwargs):
        """Compute the transition rates for the given entity (either 'wasting' or 'cause')."""
        # We need to match transition count with person-time in its from_state. We do this by
        # renaming the entity_state column in state_person_time df, and adding from_state to strata.
        transition_count = data[f"{entity}_transition_count"]
        state_person_time = data[f"{entity}_state_person_time"].rename(
            columns={f"{entity}_state": "from_state"}
        )
        strata = vop.list_columns(strata, "from_state")

        # Filter the numerator and denominator if requested
        if prefilter_query is not None:
            transition_count = transition_count.query(prefilter_query)
            state_person_time = state_person_time.query(prefilter_query)

        # Broadcast numerator over transition (and redundantly, to_state) to get the transition rate across
        # each arrow separately. Without this broadcast, we'd get the sum of all rates out of each state.
        kwargs["numerator_broadcast"] = vop.list_columns(
            "transition",
            "to_state",
            kwargs.get("numerator_broadcast"),
            df=transition_count,
            default=[],
        )
        # Divide to compute the transition rates
        transition_rates = vop.ratio(
            transition_count, state_person_time, strata=strata, **kwargs
        ).assign(measure="transition_rate")
        return transition_rates

    def get_relative_risk(
        data, measure, outcome, strata, factor, reference_category, prefilter_query=None
    ):
        """
        `measure` is one of 'prevalence', 'transition_rate', or 'mortality_rate'.
            Each of these has a different type of table for the numerator (person time, transition_count, or deaths).
        `outcome` is passed to either get_transition_rates or get_prevalence,
            and represents the outcome for which we want to compute the relative risk (e.g. 'stunting_state',
            'wasting_state', or a stratification variable for measure=='prevalence', or 'wasting' or 'cause' for
            measure=='transition_rate', or 'cause'??? or 'death'??? or None??? or cause_name???
            for measure=='mortality_rate').
            Note that `outcome` may be sort of a "meta-description" of the outcome we're interested in,
            with the actual outcome being one or more items described by this variable (e.g. the specific
            stunting or wasting categories, specific wasting state or cause state transitions, or deaths from
            a specific cause).
        `factor` is the risk factor or other stratifying variable for which we want to compute the relative risk
            (e.g. x_factor, sq_lns, stunting_state, wasting_state).
        `reference_category` is the factor category to put in the denominator to use as a reference for computing
            relative risks (e.g. the TMREL). The numerator will be broadcast over all remaining categories.
        """
        if measure == "prevalence":
            get_measure = get_prevalence
            ratio_strata = vop.list_columns(strata, outcome)
        elif measure == "transition_rate":
            get_measure = get_transition_rates
            ratio_strata = vop.list_columns(strata, "transition", "from_state", "to_state")
        elif (
            measure == "mortality_rate"
        ):  # Or burden_rate, and then pass 'death', 'yll', or 'yld' for outcome
            #         get_measure = get_rates # or get_burden_rates
            #         ratio_strata = vop.list_columns(strata, ???)
            raise NotImplementedError(
                "relative mortality rates have not yet been implemented"
            )
        else:
            raise ValueError(f"Unknown measure: {measure}")
        # Add risk factor to strata in order to get prevalence or rate in different risk factor categories
        measure_df = get_measure(
            data, outcome, vop.list_columns(strata, factor), prefilter_query
        )
        numerator = measure_df.query(f"{factor} != '{reference_category}'").rename(
            columns={f"{factor}": f"numerator_{factor}"}
        )
        denominator = measure_df.query(f"{factor} == '{reference_category}'").rename(
            columns={f"{factor}": f"denominator_{factor}"}
        )
        relative_risk = vop.ratio(
            numerator,
            denominator,
            ratio_strata,  # Match outcome categories to compute the relative risk
            numerator_broadcast=f"numerator_{factor}",
            denominator_broadcast=f"denominator_{factor}",
        ).assign(
            measure="relative_risk"
        )  # Or perhaps I should be more specific, i.e. "prevalence_ratio" or "rate_ratio"
        return relative_risk


########################
#### Module methods ####
