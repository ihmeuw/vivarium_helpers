import pandas as pd
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
        **kwargs,
    ):
        super().__init__(mapping, **kwargs)
        self.ops = VPHOperator(
            value_col,
            draw_col,
            scenario_col,
            measure_col,
            index_cols
        )

    def find_person_time_tables(self, colnames, exclude=None):
        """Generate person-time table names in this VPHResults object
        that contain the specified column names, excluding the
        specified table names.
        """
        colnames = set(list_columns(colnames))
        exclude = list_columns(exclude, default=[])
        # Create a generator of table names
        table_names = (
            table_name for table_name, table in self.items()
            if table_name not in exclude and table_name.endswith("person_time")
            and colnames.issubset(table.columns)
        )
        return table_names

    def get_person_time_table_name(self, colnames, exclude=None):
        """Return the name of a person-table that contains the
        specified columns, None if none can be found.
        """
        person_time_table_name = next(
            # Set default to None if no tables found
            find_person_time_tables(self, colnames, exclude), None)
        return person_time_table_name

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
        kwargs['numerator_broadcast'] = vop.list_columns(
            state_variable, kwargs.get('numerator_broadcast'), default=[])
        # Determine columns we need for numerator and denominator so we can look up appropriate person-time tables
        numerator_columns = vop.list_columns(strata, kwargs['numerator_broadcast'])
        denominator_columns = vop.list_columns(strata, kwargs.get('denominator_broadcast'), default=[])
        # Define numerator
        if f"{state_variable}_person_time" in data:
            state_person_time = data[f"{state_variable}_person_time"]
        else:
            # Find a person-time table that contains necessary columns for numerator.
            # Exclude cause-state person-time because it contains total person-time multiple times,
            # which would make us over-count.
            numerator_table_name = get_person_time_table_name(data, numerator_columns, exclude='cause_state_person_time')
            state_person_time = data[numerator_table_name]
        # Find a person-time table that contains necessary columns for total person-time in the denominator.
        # Exclude cause-state person-time because it contains total person-time multiple times,
        # which would make us over-count.
        denominator_table_name = get_person_time_table_name(data, denominator_columns, exclude='cause_state_person_time')
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
            **kwargs, # Includes numerator_broadcast over state_variable
        ).assign(measure='prevalence')
        return prevalence

    def get_transition_rates(data, entity, strata, prefilter_query=None, **kwargs):
        """Compute the transition rates for the given entity (either 'wasting' or 'cause')."""
        # We need to match transition count with person-time in its from_state. We do this by
        # renaming the entity_state column in state_person_time df, and adding from_state to strata.
        transition_count = data[f"{entity}_transition_count"]
        state_person_time = data[f"{entity}_state_person_time"].rename(columns={f"{entity}_state": "from_state"})
        strata = vop.list_columns(strata, "from_state")

        # Filter the numerator and denominator if requested
        if prefilter_query is not None:
            transition_count = transition_count.query(prefilter_query)
            state_person_time = state_person_time.query(prefilter_query)

        # Broadcast numerator over transition (and redundantly, to_state) to get the transition rate across
        # each arrow separately. Without this broadcast, we'd get the sum of all rates out of each state.
        kwargs['numerator_broadcast'] = vop.list_columns(
            'transition', 'to_state', kwargs.get('numerator_broadcast'), df=transition_count, default=[])
        # Divide to compute the transition rates
        transition_rates = vop.ratio(
            transition_count,
            state_person_time,
            strata = strata,
            **kwargs
        ).assign(measure='transition_rate')
        return transition_rates

    def get_relative_risk(data, measure, outcome, strata, factor, reference_category, prefilter_query=None):
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
        if measure=='prevalence':
            get_measure = get_prevalence
            ratio_strata = vop.list_columns(strata, outcome)
        elif measure=='transition_rate':
            get_measure = get_transition_rates
            ratio_strata = vop.list_columns(strata, 'transition', 'from_state', 'to_state')
        elif measure=='mortality_rate': # Or burden_rate, and then pass 'death', 'yll', or 'yld' for outcome
    #         get_measure = get_rates # or get_burden_rates
    #         ratio_strata = vop.list_columns(strata, ???)
            raise NotImplementedError("relative mortality rates have not yet been implemented")
        else:
            raise ValueError(f"Unknown measure: {measure}")
        # Add risk factor to strata in order to get prevalence or rate in different risk factor categories
        measure_df = get_measure(data, outcome, vop.list_columns(strata, factor), prefilter_query)
        numerator = (measure_df.query(f"{factor} != '{reference_category}'")
                     .rename(columns={f"{factor}":f"numerator_{factor}"}))
        denominator = (measure_df.query(f"{factor} == '{reference_category}'")
                       .rename(columns={f"{factor}":f"denominator_{factor}"}))
        relative_risk = vop.ratio(
            numerator,
            denominator,
            ratio_strata, # Match outcome categories to compute the relative risk
            numerator_broadcast=f"numerator_{factor}",
            denominator_broadcast=f"denominator_{factor}",
        ).assign(measure='relative_risk') # Or perhaps I should be more specific, i.e. "prevalence_ratio" or "rate_ratio"
        return relative_risk
