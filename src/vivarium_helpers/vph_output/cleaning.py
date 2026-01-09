import pandas as pd


def clean_vph_output(data):
    """Reformat transformed count data to make more sense.
    Modifies data in place.
    """
    # # Make the wasting and disease transition count dataframes better
    # clean_data.update(
    #     {table_name: clean_transition_df(table)
    #      for table_name, table in data.items()
    #      if table_name.endswith('transition_count')})
    if "ylds" in data and "cause_of_disability" in data["ylds"]:
        data["ylds"].rename(columns={"cause_of_disability": "cause"}, inplace=True)


def split_measure_and_transition_columns(transition_df):
    """Separates the transition from the measure in the strings in the 'measure'
    columns in a transition count dataframe, and puts these in separate 'transition'
    and 'measure' columns.
    """
    return transition_df.assign(
        transition=lambda df: df["measure"].str.replace("_event_count", "")
    ).assign(  # Older models label this event_count
        measure="transition_count"
    )  # Name the measure 'transition_count' rather than 'event_count'


def extract_transition_states(transition_df):
    """Gets the 'from state' and 'to state' from the transitions in a transition count dataframe,
    after the transition has been put in its own 'transition' column by the `split_measure_and_transition_columns`
    function.
    """
    states_from_transition_pattern = r"^(?P<from_state>\w+)_to_(?P<to_state>\w+)$"
    # Renaming the 'susceptible_to' states is a hack to deal with the fact there's not a unique string
    # separating the 'from' and 'to' states -- it should be '__to__' instead of '_to_' or something
    states_df = (
        transition_df["transition"]
        .str.replace(
            "susceptible_to", "without"
        )  # Remove word 'to' from all states so we can split transitions on '_to_'
        .str.extract(
            states_from_transition_pattern
        )  # Create dataframe with 'from_state' and 'to_state' columns
        .apply(
            lambda col: col.str.replace("without", "susceptible_to")
        )  # Restore original state names
    )
    return states_df


# Define a function to make the transition count dataframes better
def clean_transition_df(df):
    df = split_measure_and_transition_columns(df)
    return df.join(extract_transition_states(df))
