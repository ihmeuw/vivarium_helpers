from ...utils import (AttributeMapping, constant_categorical, current_time,
                      convert_to_categorical, print_memory_usage)
from ...vph_output.measures import VPHResults
from ...vph_output.operations import VPHOperator
from ...vph_output.loading import load_draws_from_keyspace_files, load_keyspace
from ...vph_output.cleaning import extract_transition_states
from ...vph_artifact.operations import convert_to_sim_format
from . import loading, population
import pandas as pd
from codetiming import Timer

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
        self.run_type = run_type
        self.run_dirs = run_dirs
        self.batch_run_dirs = batch_run_dirs
        self.all_locations_together = all_locations_together
        self.locations = (
            locations if locations is not None else loading.LOCATIONS)
        self.project_directory = (
            project_directory if project_directory is not None
            else loading.PROJECT_DIRECTORY)

        if self.run_dirs is not None and self.batch_run_dirs is not None:
            raise ValueError(
                "Must specify exactly one of run_dirs or batch_run_dirs")

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

        # Operator to perform operations on simulation results
        self.ops = VPHOperator(location_col=True)
        # dictionary to store simulation results
        self.data = VPHResults(ops=self.ops)
        # dictionary to store artifact data
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
        # twenty_five_plus_map = {
        #     '25_plus': self.age_map['age_group'].to_list()}

        # Aggregate all age groups above the minimum testing age
        min_testing_age_group = min(loading.TESTING_ELIGIBLE_AGE_GROUPS)
        min_testing_age = int(min_testing_age_group.split('_')[0])
        older_ages_map = {
            f'{min_testing_age}_plus': [
                a for a in self.age_map['age_group']
                if a >= min_testing_age_group]
            }
        both_sexes_map = {'Both': ['Male', 'Female']}
        df = (
            df
            # .pipe(self.ops.aggregate_categories, 'age_group', twenty_five_plus_map,
            #       append=True)
            .pipe(self.ops.aggregate_categories, 'age_group', older_ages_map,
                  append=True)
            .pipe(self.ops.aggregate_categories, 'sex', both_sexes_map,
                  append=True)
        )
        return df

    def reformat_population_structure(
            self,
            population_structure,
            append_aggregates=True,
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

    def process_deaths(self, deaths):
        """Preprocess the deaths dataframe and compute averted deaths."""
        # Filter to only deaths due to AD
        deaths = deaths.query("entity=='alzheimers_disease_state'")
        # Calculate averted deaths
        averted_deaths = (
            self.ops.averted(deaths, baseline_scenario='baseline')
            .assign(measure='Averted Deaths Associated with AD')
        )
        # Do transformations
        deaths = (
            deaths
            # Rename the measure
            .assign(measure='Deaths Associated with AD')
            # Concatenate deaths with averted deaths
            .pipe(lambda df:
                # Use inner join to drop "subtracted_from" column added by
                # .averted
                pd.concat([df, averted_deaths], join='inner', ignore_index=True))
            # Assign same metric to both deaths and averted deaths
            .assign(metric='Number', disease_stage=lambda df: df['entity'])
            .pipe(convert_to_categorical)
        )
        return deaths

    def process_dalys(self, ylls, ylds):
        """Process YLLs and YLDs dataframes to get DALYs and averted DALYs.
        """
        # Filter to only YLLs and YLDs due to AD, and rename so the entity
        # is the same between the two, so that the VPHResults object will
        # add YLLs and YLDs instead of keeping them separate
        ylls = (
            ylls
            .query("entity=='alzheimers_disease_state'")
            # Choose an arbitrary diseas name
            .replace({'entity': {'alzheimers_disease_state': 'AD'}})
            # Add a sub_entity column to specify disease stage
            .assign(sub_entity='alzheimers_disease_state')
            # Assign 0 YLLs to the MCI state so that when we sum with YLDs,
            # DALYs for MCI will equal YLDs. If we didn't add these 0's, it
            # would just aggregate across disease states instead of keeping
            # them separate.
            .pipe(
                lambda df: pd.concat([df, df.assign(
                    sub_entity='alzheimers_mild_cognitive_impairment_state',
                    value=0.0
                )])
            )
            .pipe(convert_to_categorical)
        )
        ylds = (
            ylds
            .query("entity=='alzheimers_disease_and_other_dementias'")
            # Choose the same arbitrary disease name
            .replace({'entity': {'alzheimers_disease_and_other_dementias': 'AD'}})
            .pipe(convert_to_categorical)
        )
        # Create a VPHResults object to calculate DALYs
        results = VPHResults(ylls=ylls, ylds=ylds, ops=self.ops)
        # Calculate DALYs and compress
        dalys = results.get_burden('dalys').pipe(convert_to_categorical)
        # print_memory_usage(dalys, 'dalys')
        # print(dalys.dtypes)

        # Calculate averted DALYs
        averted_dalys = (
            self.ops.averted(dalys, baseline_scenario='baseline')
            .assign(measure='Averted DALYs Associated with AD')
        )
        dalys = (
            dalys
            # Rename the measure
            .assign(measure='DALYs Associated with AD')
            # Concatenate deaths with averted DALYs
            .pipe(lambda df:
                # Use inner join to drop "subtracted_from" column added by
                # .averted
                pd.concat([df, averted_dalys], join='inner', ignore_index=True))
            # Assign same metric to both DALYs and averted DALYs
            .assign(metric='Number', disease_stage=lambda df: df['sub_entity'])
            .pipe(convert_to_categorical)
        )
        return dalys

    def process_prevalence_counts(self, person_time_ad):
        """Process person-time in AD states to get prevalence counts of
        each state.
        """
        prevalence_counts = (
            person_time_ad
            # Marginalize over treatment column to get total person-time in
            # each AD state
            .pipe(self.ops.marginalize, 'treatment')
            # Assign measure and metric
            .assign(measure='Prevalence', metric='Number',
                    disease_stage=lambda df: df['sub_entity'])
            # Save memory if possible
            .pipe(convert_to_categorical)
        )
        return prevalence_counts

    def process_incidence(self, incidence_bbbm, transitions_ad):
        """Preprocess incidence data."""
        incidence_bbbm = (
            incidence_bbbm
            # Assign to_state to match with transition counts dataframe
            # after extracting to and from states from transitions
            .assign(to_state='alzheimers_blood_based_biomarker_state')
        )
        transitions_ad = (
            transitions_ad
            # Display memory usage of loaded dataframe after adding columns
            .pipe(lambda df: print_memory_usage(df, 'raw transitions') or df)
            .pipe(self.ops.marginalize, 'treatment')
            # Display memory usage of loaded dataframe after adding columns
            .pipe(lambda df: print_memory_usage(df, 'after marginalizing treatment') or df)
            # Extract to and from states from transitions, and add them in
            # new columns
            .pipe(lambda df: df.join(
                extract_transition_states(df, 'sub_entity'))
            )
            # Display memory usage of loaded dataframe after adding columns
            .pipe(lambda df: print_memory_usage(df, 'after state extraction') or df)
            .pipe(lambda df: print(df.to_state.unique()) or df)
        )
        incidence = (
            # Concatenate BBBM incidence with MCI and AD incidence
            # Inner join drops irrelevant columns and avoids NaNs
            pd.concat([incidence_bbbm, transitions_ad], join='inner')
            # Assign measure, metric, and disease stage columns
            .assign(measure='Incidence', metric='Number',
                    disease_stage=lambda df: df['to_state'])
            # Save memory (and time) if possible
            .pipe(convert_to_categorical)
        )
        return incidence

    def process_mslt_results(self, mslt_results):
        """Process the multistate life table (MSLT) results to
        concatenate with simulation results (BBBM tests and treatments)
        so we can calculate rates all together.
        """
        def convert_draws_to_integers(df):
            """If df['input_draw'] contains strings, remove the prefix
            'draw_' to convert values from, e.g., 'draw_123' to 123. If
            the column doesn't contain strings, do nothing.
            """
            draw_col = df['input_draw']
            if pd.api.types.is_string_dtype(draw_col.dtype):
                df = df.assign(
                    input_draw=draw_col.str.removeprefix('draw_')
                    .astype(self.colname_to_dtype['input_draw'])
                )
            return df

        def zero_out_medication_in_testing_scenario(df):
            """Set medication initiation to 0.0 in 'BBBM Testing Only'
            scenario, because it incorrectly had nonzero values.
            """
            df.loc[
                (df['measure']=='Medication Initiation')
                & (df['scenario']=='BBBM Testing Only'), 'value'] = 0.0
            return df
        # Need to map beautified names back to sim output names to
        # process together with sim results
        column_name_map = self.get_column_name_map(inverse=True)
        mslt_results = (
            mslt_results
            .rename(columns=column_name_map)
            .pipe(convert_draws_to_integers)
            .query(f"input_draw in {self.draws}")
            .replace(
                {'measure':
                 {'BBBM False Positive Tests': 'Positive BBBM Tests',
                  'Improper Medication Uses': 'Medication Initiation'}})
            .pipe(zero_out_medication_in_testing_scenario)
            # Scale down to simulation scale to match sim outputs since
            # we'll be scaling back up later
            .pipe(self.scale_to_simulation)
            .pipe(convert_to_categorical)
            # TODO: Maybe also fill in baseline scenario with 0s?
        )
        return mslt_results

    def process_bbbm_tests(self, bbbm_tests, mslt_results):
        """Concatenate all BBBM tests with positive BBBM tests and
        preprocess for final outputs.
        """
        # Filter out counts of 'not_tested' (all 0s)
        # Also filter to age groups where tests are nonzero
        bbbm_tests = bbbm_tests.query(
            "bbbm_test_results != 'not_tested'"
            " and age_group in @loading.TESTING_ELIGIBLE_AGE_GROUPS"
        )
        # Add up positive and negative tests to get total BBBM tests
        total_bbbm_tests = (
            self.ops.marginalize(bbbm_tests, 'bbbm_test_results')
            .assign(measure='BBBM Tests', disease_stage='Preclinical AD')
        )
        # Get counts of positive tests
        positive_bbbm_tests = (
            bbbm_tests
            .query("bbbm_test_results == 'positive'")
            .assign(
                measure='Positive BBBM Tests', disease_stage='Preclinical AD')
        )
        # Filter out treatment results from preprocessed MSLT output to
        # get counts of tests among susceptible population
        susceptible_bbbm_tests = mslt_results.query(
            "measure in ['BBBM Tests', 'Positive BBBM Tests']")

        # Concatenate total tests with positive tests and tests among
        # susceptible
        bbbm_tests = (
            # inner join drops 'bbbm_test_results' column which has been
            # marginalized out of the total tests dataframe, as well as
            # 'artifact_path', 'entity', and 'entity_type' columns,
            # which are not present in MSLT results
            pd.concat(
                [total_bbbm_tests,
                 positive_bbbm_tests,
                 susceptible_bbbm_tests],
                join='inner', ignore_index=True)
            .assign(metric='Number')
            .pipe(convert_to_categorical)
        )
        return bbbm_tests

    def process_csf_pet_tests(self, csf_pet_tests):
        """Preprocess CSF and PET test counts."""
        # Filter out 'not_tested' and 'bbbm' among those eligible for
        # CSF/PET testing, and assign measure column
        csf_pet_tests = (
            csf_pet_tests
            .query("testing_state in ['csf', 'pet']")
            .assign(measure=lambda df: df['testing_state'].map(
                lambda s: f'{s.upper()} Tests'))
        )
        # Compute averted tests and assign measure
        averted_csf_pet_tests = (
            self.ops.averted(csf_pet_tests, baseline_scenario='baseline')
            .assign(measure=lambda df: df['measure'].map(
                lambda s: f'Averted {s}'))
        )
        # Concatenate and add metric column
        csf_pet_tests = (
            # Inner join drops 'subtracted_from' column added by
            # .averted function
            pd.concat([csf_pet_tests, averted_csf_pet_tests],
                      join='inner', ignore_index=True)
            .assign(metric='Number', disease_stage='MCI due to AD')
            .pipe(convert_to_categorical)
        )
        return csf_pet_tests

    def process_treatments(self, treatments, mslt_results):
        """Preprocess treatment counts, concatenating sim results with
        MSLT results.
        """
        # Filter MSLT results to treatments among susceptible population
        # TODO: Fill in age group 80-84 with 0s to match age groups for
        # sim results, and so that we don't get NaNs when computing
        # rates?
        susceptible_treatments = mslt_results.query(
            "measure=='Medication Initiation'")

        # Check measure column to determine whether we're working with
        # treatment transition counts or treatment duration
        treatments_measure = treatments['measure'].unique()
        assert len(treatments_measure) == 1, \
            f"Expected only one unique measure in treatments dataframe: {treatments_measure}"
        treatments_measure = treatments_measure[0]

        # Depending on which dataframe was passed, set the correct
        # filters, and define an appropriate function to assign
        # 'Medication Completion' vs. 'Medication Discontinuation' based
        # on the sub_entity column
        if treatments_measure == 'transition_count':
            # Treatments dataframe is treatment transition counts
            # Filter to transitions corresponding to starting treatment and
            # to age groups where treatment is nonzero
            filter_query = loading.FINAL_RESULTS_FILTER_QUERIES[
                'transition_count_treatment']
            def assign_completion_vs_discontinuation(df):
                df = df.assign(measure=df['sub_entity'].replace(
                    {'waiting_for_treatment_to_full_effect_long':
                        'Medication Completion',
                     'waiting_for_treatment_to_full_effect_short':
                        'Medication Discontinuation'}))
                return df
        elif treatments_measure == 'treatment_duration_count':
            # Treatments dataframe is treatment duration
            filter_query = loading.FINAL_RESULTS_FILTER_QUERIES[
                'treatment_duration']
            def assign_completion_vs_discontinuation(df):
                df = (df.pipe(self.ops.aggregate_categories, 'sub_entity',
                            {'Medication Completion': 9,
                             'Medication Discontinuation': range(1, 9)})
                    .drop(columns='measure')
                    .rename(columns={'sub_entity': 'measure'})
                )
                return df
        else:
            raise ValueError(
                f"Unexpected measure in treatments dataframe: {treatments_measure}")

        treatments = (
            treatments
            .query(filter_query)
            .pipe(assign_completion_vs_discontinuation)
            .assign(disease_stage='Preclinical AD')
            .pipe(self.ops.aggregate_categories, 'measure',
                  {'Medication Initiation':
                    ['Medication Completion', 'Medication Discontinuation']},
                    append=True)
            .pipe(lambda df: pd.concat(
                [df, susceptible_treatments], join='inner', ignore_index=True))
            .pipe(convert_to_categorical)
        )
        return treatments

    def scale_to_simulation(self, measure):
        """Multiply the values in the `measure` dataframe by the values in
        `model_scale`, matching location and draw, and broadcasting across
        other columns in `measure`. This converts the value of the measure
        in the real-world population to a scaled-down version on the
        same scale as the simulation.
        """
        measure = self.ops.value(measure)
        # NOTE: Reindexing preserves categoricals (in location column), but
        # results in all NaN's for some reason
        model_scale = self.ops.value(self.model_scale)#.reindex(measure.index)
        # scaled_measure = measure.divide(model_scale, axis=0).reset_index()
        scaled_measure = (measure * model_scale).reset_index()
        #.dropna() # Alternative to filtering draws above
        return scaled_measure

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
            person_time=None,
            # Default stratifications: ['event_year', 'age_group', 'sex']
            stratifications=None,
            append=False,
        ):
        """Divide a measure by person-time to get a rate."""
        if person_time is None:
            person_time = self.person_time
        # Filter person-time to age groups present in measure dataframe
        # to avoid NaNs when dividing
        person_time = person_time.query(
            f"age_group in {list(measure['age_group'].unique())}")
        if stratifications is not None:
            measure = self.ops.stratify(measure, stratifications)
            # NOTE: This may take several seconds to run, hence only runs if
            # stratifications are explicitly passed in
            person_time = self.ops.drop_index('scenario').stratify(
                person_time, stratifications)

        # Divide measure by total person-time to get rate NOTE: I'm not
        # using the ops.ratio function because, in addition to
        # specifying the stratifications, it would require explicitly
        # broadcasting over columns we need to keep, like 'measure',
        # whereas using ops.value automatically keeps all columns.
        rate = (self.ops.value(measure) / self.ops.value(person_time)
                ).reset_index()

        # Assign the 'metric' column to 'Rate', using a dtype that also
        # includes 'Number'
        metric_dtype = pd.CategoricalDtype(['Number', 'Rate'])
        rate = rate.assign(
                metric=constant_categorical('Rate', len(rate), metric_dtype))
        if append:
            # Concatenate original measure DataFrame with rates
            measure = measure.assign(
                metric=constant_categorical(
                    'Number', len(measure), metric_dtype))
            result = pd.concat([measure, rate], ignore_index=True)
        else:
            # Just return the calculated rates
            result = rate
        return result

    def get_column_name_map(
            self,
            # disease_stage is a placeholder we use in case there is no
            # such column; the actual column name depends on the dataset
            disease_stage_column='disease_stage',
            inverse=False,
        ):
        """Get the default mapping of sim output columns to final
        "beautified" column names, or the inverse of this map (for
        conforming MSLT output to sim output before concatenating and
        computing rates).
        """
        column_name_map = {
            'event_year': 'Year',
            'age_group': 'Age',
            'location': 'Location',
            'sex': 'Sex',
            'scenario': 'Scenario',
            'measure': 'Measure',
            'metric': 'Metric',
            disease_stage_column: 'Disease Stage',
            'input_draw': 'Draw',
            'value': 'Value',
            'mean': 'Mean',
            'lower': '95% UI Lower',
            'upper': '95% UI Upper',
        }
        if inverse:
            # Map beautified columns back to standard sim columns
            column_name_map = {v: k for k, v in column_name_map.items()}
        return column_name_map

    @Timer(name='SummarizingTimer', initial_text=True)
    def summarize_and_beautify(
            self,
            df,
            # By default, assume disease stage is stored in
            # disease_stage column, but allow passing a different column
            disease_stage_column='disease_stage',
        ):
        """Append rates, scale to real-world, summarize, rename columns,
        filter to desired columns, and put them in the right order.
        """
        # Default column name map
        column_name_map = self.get_column_name_map(disease_stage_column)
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
            # Append rows for "all ages" and "both sexes"
            .pipe(self.append_aggregate_categories)
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
