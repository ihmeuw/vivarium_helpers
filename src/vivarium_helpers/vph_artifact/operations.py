"""Module for performing operations on Artifacts for Vivarium Public
Health simulations.
"""
from ..utils import constant_categorical

def convert_to_sim_format(
        df,
        draws=None,
        measure=None,
        age_map=None,
        colname_to_dtype=None,
    ):
    """Convert artifact data to a format compatible with sim output."""
    def assign_age_group(df, age_map):
        """Convert age_start/age_end to age_group using age_map."""
        if (age_map is not None
            and 'age_start' in df
            and 'age_end' in df
            ):
            age_map = age_map.set_index(['age_start', 'age_end'])['age_group']
            df = (
                df
                .join(age_map, on=['age_start', 'age_end'])
                .drop(columns=['age_start', 'age_end'])
            )
        return df

    # Use an empty datatype dict if None was passed
    colname_to_dtype = colname_to_dtype or {}

    # Do transformations
    new_df = (
        df
        # Filter to specified draws
        .pipe(lambda df: df[[f'draw_{d}' for d in draws]]
              if draws is not None else df)
        .rename_axis(columns='input_draw')
        # Convert draws to integers
        .rename(columns=lambda s: int(s.removeprefix('draw_')))
        # Stack draws to index
        .stack()
        # .sort_index()
        .rename('value')
        .rename_axis(index={'year_start': 'event_year'})
        .reset_index()
        # Drop the year_end column if it exists
        .drop(columns='year_end', errors='ignore')
        # Assign age group if possible
        .pipe(assign_age_group, age_map)
        # Assign measure column if specified
        .pipe(lambda df:
              df.assign(measure=constant_categorical(measure, len(df)))
              if measure is not None else df)
        # Convert datatypes, including measure column and age_group
        # column if they are specified
        .pipe(lambda df: df.astype(
            {c: dtype for c, dtype
             in colname_to_dtype.items() if c in df}
        ))
    )
    return new_df
