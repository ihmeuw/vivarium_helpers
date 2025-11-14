"""Module for doing calculations using the population data stored
in the Artifact.
"""
from .loading import load_artifact_data
from enum import Enum
from pandas import DataFrame

POPULATION_PER_SEED = 20_000
NUM_SEEDS_V_AND_V = 5
NUM_SEEDS_FINAL = 100

class RunType(Enum):
    V_AND_V = 1
    FINAL = 2

def get_initial_simulation_population(run_type: RunType) -> int:
    """Return the total size of a simulation population, for either a
    V&V run or a final run.
    """
    if run_type == RunType.V_AND_V:
        num_seeds = NUM_SEEDS_V_AND_V
    elif run_type == RunType.FINAL:
        num_seeds =  NUM_SEEDS_FINAL
    else:
        raise ValueError(
            f"Must pass {RunType.V_AND_V} or {RunType.FINAL} for run_type")
    return num_seeds * POPULATION_PER_SEED

def get_initial_real_world_population(
       population_structure: DataFrame|None = None,
       initial_prevalence: DataFrame|None = None,
       # Start year of the simulation -- it was 2025, but then we
       # switched to 2022
       start_year: int = 2022,
       filter_terms: list|None = None,
       location_to_artifact_path: dict|None = None,
) -> DataFrame:
    """Calculate size of the real-world population that we are
    simulating, stratified by location, year, sex, and age group. This
    is obtained by multiplying the stratified forecasted population in
    each location (the "population structure") by the prevalence of all
    AD disease states combined (preclinical + MCI + AD-dementia), which
    is stored in the 'population.scaling_factor' key of the artifact.
    """
    # Check whether we were given enough data to do anything
    if location_to_artifact_path is None:
        if population_structure is None or initial_prevalence is None:
            raise ValueError(
                "Must provide either two dataframes or a dictionary"
                " mapping locations to artifact paths")
    # Load data from artifact if dataframes weren't passed in
    if population_structure is None:
        # This is the number of people in each demographic group in each
        # year -- these numbers come from the FHS population forecasts
        population_structure = load_artifact_data(
            'population.structure', filter_terms, location_to_artifact_path)
    if initial_prevalence is None:
        # For each demographic group, the "population scaling factor" is
        # the ratio of the real-world population that we want to
        # simulate in that group to the total number of people in that
        # group. For Model 4 and above, this equals the initial
        # prevalence of all AD disease states combined (preclinical +
        # MCI + AD-dementia), since we are modeling the population of
        # people with any stage of AD. Note that this is defined for the
        # population at the beginning of the simulation, so there is
        # only one year of data.
        #
        # NOTE: This data has two age groups, 95-100 and 100-105,
        # instead of the single age group 95-125 that's in the
        # population structure. I'm not sure why. I'm going to drop the
        # 100-105 age group and match the 95-100 age group with the
        # 95-125 age group from above
        initial_prevalence = load_artifact_data(
            'population.scaling_factor', filter_terms, location_to_artifact_path)
    # There should be only one year of data for the initial prevalence
    years = initial_prevalence.index.unique('year_start')
    assert len(years) == 1, 'Unexpected years for initial prevalence!'
    year = years[0]
    # Use the specified start year for the population structure,
    # regardless of what single year is stored in the initial
    # prevalence. Rename year_start and year_end in initial_prevalence
    # to match the start year in the population structure.
    initial_prevalence = (
        initial_prevalence
        .rename({year: start_year}, level='year_start')
        # NOTE: Only works if year_end = year_start + 1
        .rename({year + 1: start_year + 1}, level='year_end')
    )
    initial_prevalence_counts = (
        population_structure
        .query("year_start==@start_year")
        # Change end of oldest age group to match prevalence data
        .rename({125.0: 100.0}, level='age_end')
        * initial_prevalence
    ).dropna() # Drop age groups we don't have in sim
    return initial_prevalence_counts

def calculate_model_scale(
        initial_simulation_population: int|RunType,
        initial_real_world_population: DataFrame|None = None,
        location_to_artifact_path: dict|None = None
    ) -> DataFrame:
    """Calculate the model scale for a simulation run, which is the
    ratio of the real-world population we are modeling to the initial
    population in the simulation. The result will be a single number for
    each location and input draw and is returned in Artifact format,
    with one row for each location and one column for each draw.
    """
    if (initial_real_world_population is None
        and location_to_artifact_path is None):
        raise ValueError(
            "Must provide either the initial real-world population dataframe"
            " or a dictionary mapping locations to artifact paths")
    if isinstance(initial_simulation_population, RunType):
        initial_simulation_population = get_initial_simulation_population(
            initial_simulation_population)
    if initial_real_world_population is None:
        initial_real_world_population = get_initial_real_world_population(
            location_to_artifact_path=location_to_artifact_path)
    # Sum over age groups to get real-world population in each location
    total_initial_real_world_pop = (
        initial_real_world_population.groupby('location').sum())
    # Model scale is the ratio of our simulated population to the real-world
    # population at time 0
    model_scale = (
        initial_simulation_population / total_initial_real_world_pop)
    # This format (draws horizontally as column names, as strings) is
    # compatible with Artifacts
    return model_scale
