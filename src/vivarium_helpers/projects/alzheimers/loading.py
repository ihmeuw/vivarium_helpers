"""Code used to load files for the CSU Alzheimer's project."""
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from vivarium import Artifact
from ...utils import convert_to_categorical, print_memory_usage

# Project directory
project_dir = Path('/mnt/team/simulation_science/pub/models/vivarium_csu_alzheimers/')

# For testing: Run directory containing model 8.3 results for all
# locations
model_run_subdir = 'results/abie_consistent_model_test/united_states_of_america/2025_10_28_08_55_05/'

# Results directory for model 8.3, for testing
results_dirs = project_dir / model_run_subdir / 'results/'

# Artifact for models 8.3 - 8.7
artifact_model_number = '8.3'

locations = [
    'United States of America',
    'Brazil',
    'China',
    'Germany',
    'Israel',
    'Japan',
    'Spain',
    'Sweden',
    'Taiwan (Province of China)',
    'United Kingdom',
]

def get_results_and_artifact_dicts(
        locations, results_dirs, artifact_model_number, project_dir):

    match results_dirs:
        case str() | Path():
            # Option 1: All locations concatenated in one results
            # directory
            location_to_results_dir = {'all': results_dirs}
        case list():
            # Option 2: One results directory per location
            location_to_results_dir = {
                loc: path for loc, path in zip(locations, results_dirs)}

    location_to_artifact_subdir = {
        loc: loc.lower().replace(' ', '_') for loc in locations}
    artifact_subpaths = [
        f'artifacts/model{artifact_model_number}/' + subdir + '.hdf'
        for subdir in location_to_artifact_subdir.values()]

    location_to_artifact_path = {
        # Make sure artifact directory is stored as a string, not a Path
        # object, since it'll be stored as a string in the simulation
        # output, and we'll need to reverse this dict to map from
        # directories to locations
        loc: str(project_dir / subpath) for loc, subpath
        in zip(locations, artifact_subpaths)}

    return location_to_results_dir, location_to_artifact_path

def get_column_dtypes(locations):
    """Create a dictionary mapping column names to datatypes. We specify
    integer dtypes and ordered Categoricals to convert years from
    strings to ints, save memory, and have a standardized ordering of
    column entries.
    """
    # Order locations lexicographically
    location_dtype = pd.CategoricalDtype(sorted(locations), ordered=True)

    # int16 ranges from -32768 to 32767 (I think), which is sufficient to
    # represent all years 2025-2100. uint8 only goes from 0 to 255, which is
    # too small.
    year_dtype = 'int16'

    # Store draws as ints instead of categoricals since we'll be
    # concatenating different draws from different results directories
    input_draw_dtype = 'int16'

    # Order sexes alphabetically
    sex_dtype = pd.CategoricalDtype(['Female', 'Male'], ordered=True)

    # Order age groups chronologically
    age_groups = [f'{age}_to_{age + 4}' for age in range(25, 95, 5)] + ['95_plus']
    age_group_dtype = pd.CategoricalDtype(age_groups, ordered=True)

    # Order scenarios by complexity
    scenarios = ['baseline', 'bbbm_testing', 'bbbm_testing_and_treatment']
    scenario_dtype = pd.CategoricalDtype(scenarios, ordered=True)

    # Map column names to dtypes
    colname_to_dtype = {
        'location': location_dtype,
        'event_year': year_dtype,
        'age_group': age_group_dtype,
        'sex': sex_dtype,
        'scenario': scenario_dtype,
        'input_draw': input_draw_dtype,
    }
    return colname_to_dtype

#### Generate global dictionaries to use as defaults for loading data ####

# Create location-to-directory dictionaries
location_to_results_dir, location_to_artifact_path = get_results_and_artifact_dicts(
    locations, results_dirs, artifact_model_number, project_dir
)

# Create column-to-datatype dictionary
colname_to_dtype = get_column_dtypes(locations)

#### Functions to load data ####

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

def load_measure_from_batch_runs(
        measure,
        batch_results_dirs,
        locations=locations,
        # With n = 1 this loads all 10 locations at once, which may use
        # too much memory. With n > 1, this loads the locations in n
        # groups of approximate size 10 / n before aggregating seeds,
        # which is slightly slower but uses less memory.
        n_location_groups=1,
        filter_burn_in_years=True,
        artifact_model_number=artifact_model_number,
        colname_to_dtype=colname_to_dtype,
        project_dir=project_dir,
        **kwargs
    ):
    """Load data from multiple batch runs, aggregate random seeds, and
    concatenate.
    """
    # aggregate seeds by default, and warn if False was passed
    if not kwargs.setdefault('aggregate_seeds', True):
        # Documentation for setdefault: If key is in the dictionary,
        # return its value. If not, insert key with a value of default
        # and return default.
        print("Warning: Not aggregating seeds, which may require lots of memory")
    if filter_burn_in_years:
        # Filter out years before 2025 because for model 8.4, years
        # 2022-2024 are for burn-in
        year_filter = ('event_year', '>=', '2025')
        # Add the year filter to the user filters
        user_filters = kwargs.get('filters') # Defaults to None
        kwargs['filters'] = add_parquet_AND_filter(year_filter, user_filters)
    dfs = []
    print(kwargs.get('filters'))
    for results_dir in batch_results_dirs:
        print(results_dir)
        for i in range(n_location_groups):
            # Group locations into n groups. This seems to work for any
            # n and splits as evenly as possible, front-loading with
            # larger groups at the beginning.
            location_group = locations[i::n_location_groups]
            # print(location_group)
            location_to_results_dir, location_to_artifact_path = get_results_and_artifact_dicts(
                location_group, results_dir, artifact_model_number, project_dir
            )
            print(location_to_artifact_path)
            df = load_sim_output(
                measure, location_to_results_dir, location_to_artifact_path, colname_to_dtype, **kwargs
            )
            print_memory_usage(df, 'after aggregating seeds and converting dtypes')
            dfs.append(df)
    measure_df = pd.concat(dfs, ignore_index=True)
    print_memory_usage(measure_df, 'total')
    measure_df = measure_df.astype(colname_to_dtype)
    print_memory_usage(measure_df, 'after enforcing dtypes')
    return measure_df

def add_parquet_AND_filter(new_filter, existing_filters):
    """Add a filter to an existing list of parquet filters in
    disjunctive normal form (DNF). The new filter will be combined with
    the ALL existing filters via conjunction (AND), so adding the new
    filter will result in a stricter filtering criterion.
    """
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
