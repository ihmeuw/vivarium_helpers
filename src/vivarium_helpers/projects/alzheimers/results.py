from ...vph_output.measures import VPHResults
from ...vph_output.operations import VPHOperator
from ...vph_output.loading import load_draws_from_keyspace_files, load_keyspace
from ...vph_artifact.operations import convert_to_sim_format
from . import loading
import pandas as pd

class AlzheimersResultsProcessor:
    """Class for processing results of the Alzheimer's sim."""

    def __init__(
            self,
            artifact_model_number,
            run_type,
            locations=None,
            run_dirs=None,
            batch_run_dirs=None,
            all_locations_together=True,
            project_directory=None,
    ):
        self.artifact_model_number = artifact_model_number
        self.run_type=run_type
        self.run_dirs = run_dirs
        self.batch_run_dirs = batch_run_dirs
        self.all_locations_together = all_locations_together
        self.locations = (
            locations if locations is not None else loading.LOCATIONS)
        self.project_directory = (
            project_directory if project_directory is not None
            else loading.PROJECT_DIRECTORY)

        if self.run_dirs is not None and self.batch_run_dirs is not None:
            raise ValueError("Must specify exactly one of run_dirs or batch_run_dirs")

        if self.batch_run_dirs is not None:
            # Each batch contains different draws, so concatenate them together
            self.draws = load_draws_from_keyspace_files(self.batch_run_dirs)
        elif self.run_dirs is not None:
            locs = 'all' if all_locations_together else self.locations
            self.location_to_results_dir = (
                loading.get_location_results_dict(self.run_dirs, locs))
            # All model runs contain the sam draws, so just read them
            # from the first run
            self.draws = load_keyspace(self.run_dirs[0])['input_draw']
        else:
            raise ValueError("run_dirs and batch_run_dirs cannot both be None")

        if self.run_dirs is not None:
            self.location_to_results_dir = (
                loading.get_location_results_dict(self.run_dirs))
        self.location_to_artifact_path = loading.get_location_artifact_dict(
            self.locations, self.artifact_model_number)
        self.age_map = loading.get_age_map(
            self.location_to_artifact_path[self.locations[0]])
        self.colname_to_dtype = loading.get_column_dtypes(self.locations)
        self.ops = VPHOperator(location_col=True)
        self.data = VPHResults(ops=self.ops)

    def append_aggregate_categories(self, df):
        # NOTE: If age_group column is Categorical, calling .unique() also
        # returns a Categorical, which must be explicitly converted to a
        # list in order for the _ensure_iterable function to work
        all_ages_map = {'all_ages': list(df['age_group'].unique())}
        both_sexes_map = {'Both': ['Male', 'Female']}
        df = (
            df
            .pipe(self.ops.aggregate_categories, 'age_group', all_ages_map,
                  append=True)
            .pipe(self.ops.aggregate_categories, 'sex', both_sexes_map,
                  append=True)
        )
        return df

    def reformat_population_structure(
            self,
            population_structure,
            draws=None,
            append_aggregates=False,
            measure='person_time',
            age_map=None,
            colname_to_dtype=None,
    ):
        pop_structure = (
            population_structure
            # Filter to ages and years used in the sim
            .query("age_start >= 25 and year_start >= 2025")
            .pipe(convert_to_sim_format, self.draws, measure, self.age_map, self.colname_to_dtype)
            # .assign(measure=lambda df: constant_categorical('person_time', len(df)))
        )
        # Copy 2050 population forward through 2100
        pop_2050 = pop_structure.query("event_year == 2050")
        future_years = [pop_2050.assign(event_year=y) for y in range(2051, 2101)]
        pop_structure = pd.concat([pop_structure, *future_years], ignore_index=True)
        if append_aggregates:
            pop_structure = self.append_aggregate_categories(pop_structure)
        return pop_structure

