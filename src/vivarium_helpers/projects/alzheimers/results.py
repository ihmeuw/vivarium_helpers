from ...utils import (AttributeMapping, constant_categorical, current_time,
                      convert_to_categorical)
from ...vph_output.measures import VPHResults
from ...vph_output.operations import VPHOperator
from ...vph_output.loading import load_draws_from_keyspace_files, load_keyspace
from ...vph_artifact.operations import convert_to_sim_format
from . import loading, population
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

        self.location_to_artifact_path = loading.get_location_artifact_dict(
            self.locations, self.artifact_model_number)
        self.age_map = loading.get_age_map(
            self.location_to_artifact_path[self.locations[0]])
        self.colname_to_dtype = loading.get_column_dtypes(self.locations)
        self.initial_simulation_population = (
            population.get_initial_simulation_population(self.run_type))

        self.ops = VPHOperator(location_col=True)
        self.data = VPHResults(ops=self.ops)
        self.art_data = AttributeMapping()

    def load_population_data(
            self,
            append_aggregates=False,
            person_time_measure=None,
            model_scale_measure=None,
        ):
        """Load population structure and initial prevalence from the
        artifact, and compute the model scale.
        """
        self.art_data.population_structure = loading.load_artifact_data(
            'population.structure', None, self.location_to_artifact_path)
        self.art_data.all_states_initial_prev = loading.load_artifact_data(
            'population.scaling_factor', None, self.location_to_artifact_path)
        self.art_data.all_states_initial_prev_counts = (
            population.get_initial_real_world_population(
                self.art_data.population_structure,
                self.art_data.all_states_initial_prev)
        )
        self.art_data.model_scale = population.calculate_model_scale(
            self.initial_simulation_population,
            self.art_data.all_states_initial_prev_counts,
        )
        self.model_scale = convert_to_sim_format(
            self.art_data.model_scale, self.draws, model_scale_measure,
            self.age_map, self.colname_to_dtype)
        self.person_time = self.reformat_population_structure(
            self.art_data.population_structure,
            append_aggregates, person_time_measure
        )

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
            append_aggregates=False,
            measure='person_time',
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

    def load_measure_from_batch_runs(
            self,
            measure,
            save_name=None,
            batch_run_dirs=None,
            locations=None,
            # With n = 1 this loads all 10 locations at once, which may use
            # too much memory. With n > 1, this loads the locations in n
            # groups of approximate size 10 / n before aggregating seeds,
            # which is slightly slower but uses less memory.
            n_location_groups=1,
            filter_burn_in_years=True,
            artifact_model_number=None,
            colname_to_dtype=None,
            project_dir=None,
            **kwargs,
        ):
        """Load data from multiple batch runs, aggregate random seeds, and
        concatenate. Save the loaded data to this object's data
        dictionary by default, using the specified save_name. Pass
        save_name=False to return the data instead of saving.
        """
        sim_output = loading.load_measure_from_batch_runs(
            measure,
            batch_run_dirs or self.batch_run_dirs,
            locations or self.locations,
            n_location_groups,
            filter_burn_in_years,
            artifact_model_number or self.artifact_model_number,
            colname_to_dtype or self.colname_to_dtype,
            project_dir or self.project_directory,
            **kwargs,
        )
        if save_name is False:
            # Return loaded data instead of saving to dictionary
            return sim_output
        else:
            # Save to our data dictionary, using measure as the default
            # name if None was passed
            self.data[save_name or measure] = sim_output

    def scale_to_real_world(self, measure):
        """Divide the values in the `measure` dataframe by the values in
        `model_scale`, matching location and draw, and broadcasting across
        other columns in `measure`. This computes the value of the measure
        in the real-world population from the scaled-down version we get
        from the simulation.
        """
        measure = self.ops.value(measure)
        # NOTE: Reindexing preserves categoricals (in location column), but
        # results in all NaN's for some reason
        model_scale = self.ops.value(self.model_scale)#.reindex(measure.index)
        # scaled_measure = measure.divide(model_scale, axis=0).reset_index()
        scaled_measure = (measure / model_scale).reset_index()
        #.dropna() # Alternative to filtering draws above
        return scaled_measure

    def calculate_rate(
            self,
            measure,
            # Default stratifications: ['event_year', 'age_group', 'sex']
            stratifications=None,
            append=False,
        ):
        """Divide a measure by person-time to get a rate."""
        if stratifications is None:
            stratifications = ['event_year', 'age_group', 'sex']
        rate = self.ops.drop_index('scenario').ratio(
            measure,
            self.person_time,
            stratifications,
            numerator_broadcast='scenario',
            record_inputs=False,
        )
        if append:
            # Concatenate original measure DataFrame with rates, adding a
            # 'metric' column to distinguish between them
            metric_dtype = pd.CategoricalDtype(['Number', 'Rate'])
            measure = measure.assign(
                metric=constant_categorical(
                    'Number', len(measure), metric_dtype))
            rate = rate.assign(
                metric=constant_categorical('Rate', len(rate), metric_dtype))
            result = pd.concat([measure, rate], ignore_index=True)
        else:
            # Just return the calculated rates
            result = rate
        return result

    def summarize_and_beautify(
            self,
            df,
            # By default, assume disease stage is stored in sub_entity
            # column, but allow passing a different column
            disease_stage_column='sub_entity',
        ):
        """Append rates, scale to real-world, summarize, rename columns,
        filter to desired columns, and put them in the right order.
        """
        # Default column name map
        column_name_map = {
            'event_year': 'Year',
            'age_group': 'Age',
            'location': 'Location',
            'sex': 'Sex',
            'scenario': 'Scenario',
            'measure': 'Measure',
            'metric': 'Metric',
            disease_stage_column: 'Disease Stage',
            'mean': 'Mean',
            'lower': '95% UI Lower',
            'upper': '95% UI Upper',
        }
        disease_stage_name_map = {
            'alzheimers_blood_based_biomarker_state': 'Preclinical AD',
            'alzheimers_mild_cognitive_impairment_state': 'MCI due to AD',
            'alzheimers_disease_state' : 'AD Dementia'
        }
        scenario_name_map = {
            'baseline': 'Reference',
            'bbbm_testing': 'BBBM Testing Only',
            'bbbm_testing_and_treatment' : 'BBBM Testing and Treatment',
        }
        column_order = [
            'Year', 'Location', 'Age', 'Sex' , 'Disease Stage' , 'Scenario',
            'Measure', 'Metric', 'Mean', '95% UI Lower', '95% UI Upper',
        ]
        current_time()
        # Do transformations
        df = (
            df
            # Filter out burn-in years before 2025
            .query('event_year >= 2025')
            # Scale to real-world values
            .pipe(self.scale_to_real_world)
            # # Append rows for "all ages" and "both sexes"
            # .pipe(append_aggregate_categories, ops)
            # Calculate and append rates
            .pipe(self.calculate_rate, append=True)
            # Compress data if possible
            .pipe(convert_to_categorical)
            .pipe(lambda df: current_time() or df)
            # Summarize data
            .pipe(self.ops.summarize_draws)
            .reset_index()
            .pipe(lambda df: current_time() or df)
            # Rename columns
            .rename(columns=column_name_map)
            .replace(
                {'Disease Stage': disease_stage_name_map,
                'Scenario': scenario_name_map})
            [column_order]
        )
        return df
