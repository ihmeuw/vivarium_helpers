"""Code used to load files for the CSU Alzheimer's project.
"""
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from vivarium import Artifact
from ...utils import convert_to_categorical, print_memory_usage

location_to_artifact_path = {
    'United States of America': '/mnt/team/simulation_science/pub/models/vivarium_csu_alzheimers/artifacts/model8.3/united_states_of_america.hdf',
    'Brazil': '/mnt/team/simulation_science/pub/models/vivarium_csu_alzheimers/artifacts/model8.3/brazil.hdf',
    'China': '/mnt/team/simulation_science/pub/models/vivarium_csu_alzheimers/artifacts/model8.3/china.hdf',
    'Germany': '/mnt/team/simulation_science/pub/models/vivarium_csu_alzheimers/artifacts/model8.3/germany.hdf',
    'Israel': '/mnt/team/simulation_science/pub/models/vivarium_csu_alzheimers/artifacts/model8.3/israel.hdf',
    'Japan': '/mnt/team/simulation_science/pub/models/vivarium_csu_alzheimers/artifacts/model8.3/japan.hdf',
    'Spain': '/mnt/team/simulation_science/pub/models/vivarium_csu_alzheimers/artifacts/model8.3/spain.hdf',
    'Sweden': '/mnt/team/simulation_science/pub/models/vivarium_csu_alzheimers/artifacts/model8.3/sweden.hdf',
    'Taiwan (Province of China)': '/mnt/team/simulation_science/pub/models/vivarium_csu_alzheimers/artifacts/model8.3/taiwan_(province_of_china).hdf',
    'United Kingdom': '/mnt/team/simulation_science/pub/models/vivarium_csu_alzheimers/artifacts/model8.3/united_kingdom.hdf',
}

location_to_results_dir = {
    'all': Path('/mnt/team/simulation_science/pub/models/vivarium_csu_alzheimers/results/abie_consistent_model_test/united_states_of_america/2025_10_28_08_55_05/results')
}

colname_to_dtype = {
    'location': pd.CategoricalDtype(categories=['Brazil', 'China', 'Germany', 'Israel', 'Japan', 'Spain',
                   'Sweden', 'Taiwan (Province of China)', 'United Kingdom',
                   'United States of America'], ordered=True),
    'event_year': 'int16',
    'age_group': pd.CategoricalDtype(categories=['25_to_29', '30_to_34', '35_to_39', '40_to_44', '45_to_49',
                   '50_to_54', '55_to_59', '60_to_64', '65_to_69', '70_to_74',
                   '75_to_79', '80_to_84', '85_to_89', '90_to_94', '95_plus'], ordered=True),
    'scenario': pd.CategoricalDtype(categories=['baseline', 'bbbm_testing', 'bbbm_testing_and_treatment'], ordered=True),
    'input_draw': 'int16',
}


def load_artifact_data(
    key,
    filter_terms=None,
    location_to_artifact_path=location_to_artifact_path,
):
    dfs = {} # dict to map locations to artifact data
    for location, path in location_to_artifact_path.items():
        art = Artifact(path, filter_terms)
        # Check to make sure location matches artifact
        art_locations = art.load('metadata.locations')
        assert len(art_locations) == 1 and art_locations[0] == location, \
            f'Unexpected locations in artifact: {location=}, {art_locations=}'
        df = art.load(key)
        dfs[location] = df
    if all('location' in df.index.names for df in dfs.values()):
        data = pd.concat(dfs.values())
    else:
        data = pd.concat(dfs, names=['location', *df.index.names])
    return data


def load_sim_output(
        measure,
        results_dict=location_to_results_dir,
        # Pass None to skip filtering locations (when None, must also
        # pass assign_location=False or raw=True)
        location_to_artifact_path=location_to_artifact_path,
        # specify dtypes of certain columns
        colname_to_dtype=colname_to_dtype,
        drop_superfluous_cols=True, # drop redundant or empty columns
        # Sets the 'read_dictionary' key of kwargs, which is passed to
        # pyarrow.parquet.read_table()
        force_parquet_dictionaries=True,
        force_pandas_categoricals=True,
        aggregate_seeds=True,
        assign_location=True,
        raw=False, # Overrides other parameters if True
        **kwargs, # keyword args to pass to .read_parquet
    ):
    """Load simulation output from .parquet files for all locations,
    optionally reducing the size of the data when possible. Returns
    concatenated outputs with a 'location' column added.
    """
    # Override optional transformations if raw=True
    if raw:
        drop_superfluous_cols = False
        force_parquet_dictionaries = False
        force_pandas_categoricals = False
        aggregate_seeds = False
        assign_location = False

    # Determine whether results for all locations are stored in same
    # directory, or if different locations have different results
    # directories
    match location_to_results_dir:
        case {'all': _}:
            all_locations_together = True
        case _:
            all_locations_together = False

    if all_locations_together and assign_location and location_to_artifact_path is None:
        raise ValueError(
            "Must provide mapping of artifacts to locations  when" \
            " assign_location=True and all locations are concatenated" \
            " in the simulation outputs."
        )

    dfs = []
    for location, directory in results_dict.items():

        parquet_file_path = Path(directory) / f'{measure}.parquet'
        # Read the Parquet file's schema to get column names and data types
        parquet_schema = pq.read_schema(parquet_file_path)

        if (
            all_locations_together
            and location_to_artifact_path is not None
        ):
            if 'artifact_path' in parquet_schema.names:
                # Filter to locations in list
                location_filter = (
                    'artifact_path',
                    'in',
                    list(location_to_artifact_path.values()),
                )
                user_filters = kwargs.get('filters') # Defaults to None
                kwargs['filters'] = add_parquet_AND_filter(
                    location_filter, user_filters)
                # TODO: Use logging not printing
                print(location_filter)
            else:
                print("'artifact_path' column missing from parquet file."
                      " Not filtering locations.")

        if force_parquet_dictionaries:
            # Read all columns as dictionaries except those containing
            # floating point values
            kwargs['read_dictionary'] = [
                col.name for col in parquet_schema
                if not pa.types.is_floating(col.type)]

        print(kwargs.get('filters'))
        # Read the parquet file
        df = pd.read_parquet(parquet_file_path, **kwargs)
        print_memory_usage(df, 'after read_parquet')

        if drop_superfluous_cols:
            # Drop redundant columns
            for col1, col2 in [
                ('input_draw', 'input_draw_number'),
                ('entity', 'sub_entity'),
            ]:
                if (col1 in df and col2 in df and df[col1].equals(df[col2])):
                    df.drop(columns=col2, inplace=True)
            # Drop empty columns (e.g., sub-entity)
            for col in df:
                if df[col].isna().all():
                    df.drop(columns=col, inplace=True)
        if colname_to_dtype is not None:
            df = df.astype(
                # Filter to avoid KeyError
                {c: dtype for c, dtype
                 in colname_to_dtype.items() if c in df},
                 # NOTE: If copy-on-write is enabled, copy keyword is
                 # ignored
                 copy=False)
        if force_pandas_categoricals:
            convert_to_categorical(
                df, exclude_cols=colname_to_dtype or (), inplace=True)
        if aggregate_seeds:
            # Group sum values across random seeds
            groupby_cols = df.columns.difference(
                ['random_seed', 'value']).to_list()
            # Use observed=False to handle Categoricals; use
            # dropna=False in case sub_entity or some other column has
            # NaNs
            df = df.groupby(
                groupby_cols, observed=True,
                as_index=False, dropna=False)['value'].sum()
        if assign_location:
            if all_locations_together:
                # NOTE: location_to_artifact_path is guaranteed not to
                # be None because assign_location and
                # all_locations_together are both True

                # Find or create a Categorical dtype with all locations
                location_dtype = colname_to_dtype.get(
                    'location',
                    pd.CategoricalDtype(
                        sorted(location_to_artifact_path.keys()), ordered=True)
                )
                # Invert the dictionary so we can map artifact paths to
                # locations
                artifact_path_to_location = {
                    path: loc for loc, path
                    in location_to_artifact_path.items()}
                if 'artifact_path' in df:
                    df['location'] = df['artifact_path'].map(
                        artifact_path_to_location).astype(location_dtype)
                else:
                    # In case the engineers change the DataFrame format
                    # on us...
                    print("'artifact_path' column missing from DataFrame."
                          " Not assigning locations.")
            else:
                # NOTE: location_to_results_dir contains actual
                # locations as keys (not 'all') since
                # all_locations_together is False

                # Find or create a Categorical dtype with all locations
                # to avoid converting back to object dtype.
                location_dtype = colname_to_dtype.get(
                    'location',
                    pd.CategoricalDtype(
                        sorted(location_to_results_dir.keys()), ordered=True)
                )
                df['location'] = location
                df['location'] = df['location'].astype(location_dtype)
        dfs.append(df)
    # TODO: Maybe if assign_location is False and all_locations_together
    # is also False (and there is more than one location?), we should
    # return a dict mapping locations to dataframes (or just a list of
    # dataframes?) instead of concatenating, since it won't be possible
    # to filter the resulting concatenated dataframe by location...
    df = pd.concat(dfs, ignore_index=True)
    return df

def add_parquet_AND_filter(new_filter, existing_filters):
    match existing_filters:
        case None:
            # No existing filters -- create a single AND group
            filters = [new_filter]
        case list([tuple((_, _, _)), *_]):
            # Existing filters consist of one AND group -- add the new filter
            filters = [new_filter, *existing_filters]
        case list([list([tuple((_, _, _)), *_]), *_]):
            # Add the filter to each AND group in the outer OR group
            filters = [[new_filter, *and_group] for and_group in existing_filters]
        case _:
            raise ValueError(f"Malformed parquet filter: {existing_filters}")
    return filters
