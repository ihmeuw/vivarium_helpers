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
        record_dalys=True,
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
        self._clean_vph_output()
        if record_dalys:
            self.compute_and_save_dalys()

    def _clean_vph_output(self):
        """Reformat transformed count data to make more sense."""
        # TODO: Maybe implement this as a module method and
        # call it from here, since it will probably be a lot of code
        #
        # # Make the wasting and disease transition count dataframes better
        # clean_data.update(
        #     {table_name: clean_transition_df(table)
        #      for table_name, table in data.items()
        #      if table_name.endswith('transition_count')})
        if 'ylds' in self and 'cause_of_disability' in self['ylds']:
            self['ylds'].rename(
                        columns={'cause_of_disability': 'cause'}, inplace=True)

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
        dalys = self.ops.aggregate_categories(
            dalys, 'measure', {'dalys': ['ylls', 'ylds']})
        # print(f'{len(dalys)=}')
        return dalys

    def compute_and_save_dalys(self):
        self['dalys'] = self.compute_dalys()

    def table_names(self):
        return [name for name in self if name != 'ops']

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

    def get_person_time_table_name(self, colnames, exclude=None, error=False):
        """Return the name of a person-table that contains the
        specified columns. If no such table can be found, returns
        None unless `error` is True, in which case a ValueError
        is raised.
        """
        person_time_table_name = next(
            # Set default to None if no tables found
            self.find_person_time_tables(colnames, exclude), None)
        if error and person_time_table_name is None:
            raise ValueError(
                f"No person-time table found with columns {colnames}."
                f" (Excluded tables: {exclude})")
        return person_time_table_name

    def get_burden_rate(self, measure, strata, prefilter_query=None, **kwargs):
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
        burden = self[measure]
        denominator_columns = list_columns(
            strata, kwargs.get('denominator_broadcast'), default=[])
        denominator_table_name = self.get_person_time_table_name(
            denominator_columns, exclude='cause_state_person_time', error=True)
        person_time = self[denominator_table_name]
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
        ).assign(measure=f'{measure[:-1]}_rate')
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

########################
#### Module methods ####

def split_measure_and_transition_columns(transition_df):
    """Separates the transition from the measure in the strings in the 'measure'
    columns in a transition count dataframe, and puts these in separate 'transition'
    and 'measure' columns.
    """
    return (transition_df
            .assign(transition=lambda df: df['measure']
                    .str.replace('_event_count', '')) # Older models label this event_count
            .assign(measure='transition_count') # Name the measure 'transition_count' rather than 'event_count'
           )

def extract_transition_states(transition_df):
    """Gets the 'from state' and 'to state' from the transitions in a transition count dataframe,
    after the transition has been put in its own 'transition' column by the `split_measure_and_transition_columns`
    function.
    """
    states_from_transition_pattern = r"^(?P<from_state>\w+)_to_(?P<to_state>\w+)$"
    # Renaming the 'susceptible_to' states is a hack to deal with the fact there's not a unique string
    # separating the 'from' and 'to' states -- it should be '__to__' instead of '_to_' or something
    states_df = (
        transition_df['transition']
        .str.replace("susceptible_to", "without") # Remove word 'to' from all states so we can split transitions on '_to_'
        .str.extract(states_from_transition_pattern) # Create dataframe with 'from_state' and 'to_state' columns
        .apply(lambda col: col.str.replace("without", "susceptible_to")) # Restore original state names
    )
    return states_df

# Define a function to make the transition count dataframes better
def clean_transition_df(df):
    df = split_measure_and_transition_columns(df)
    return df.join(extract_transition_states(df))
