"""
Module providing functions and data structures for loading and storing (transformed) Vivarium Public Health output files
(i.e. the minimally processed .csv or .hdf tables saved in the "count_data" folder for a model).
"""

import os

import pandas as pd

from ..utils import AttributeMapping, convert_to_variable_name


class VPHOutput(AttributeMapping):
    """Implementation of the Mapping abstract base class to conveniently store transformed
    Vivarium count data tables as object attributes.

    Create a VPHOutput object from a dictionary (or other implementation of Mapping)
        that maps table names (strings) to tables (e.g. pandas DataFrames). Dictionary keys become object attributes
        that store the corresponding dictionary values. The keys (table names) must be valid Python variable names
        (this is not enforced in the constructor, but bad variable names will cause problems for you later).
    """

    @classmethod
    def from_directory(cls, directory, ext=".hdf", **kwargs):
        """Create a VivariumTransformedOutput object from the directory where the count data tables from a single
        simulation are stored.

        Example usage:
        data = VivariumTransformedOutput.from_directory('/path/to/count_data')
        data.ylls # YLL table
        data.person_time # Person-time table
        """
        return cls(load_transformed_count_data(directory, ext, **kwargs))

    @classmethod
    def from_locations_paths(cls, locations_paths, subdirectory="count_data"):
        """Create a VivariumTransformedOutput object from a dictionary that maps each location name to a
        directory that contains a subdirectory with the count data tables for that location (you can specify
        subdirectory='' if the dictionary contains paths directly to the count data). Each location name will
        become an attribute that stores another VivariumTransformedOutput object with the corresponding tables.

        Example usage:
        # Assuming the count data for 'location' is in '/path/to/location/output/count_data'
        locations_paths = {'Ethiopa': '/path/to/ethiopia/output', 'India': '/path/to/india/output'}
        data = VivariumTransformedOutput.from_locations_paths(locations_paths, subdirectory='count_data')
        data.ethiopia.ylls # YLL table for Ethiopia
        data.india.person_time # Person-time table for India
        """
        locations_count_data = load_count_data_by_location(locations_paths, subdirectory)
        return cls(
            {
                convert_to_variable_name(location.lower()): cls(count_data)
                for location, count_data in locations_count_data.items()
            }
        )

    @classmethod
    def merged_from_locations_paths(cls, locations_paths, subdirectory="count_data"):
        """Create a VivariumTransformedOutput object from a dictionary that maps each location name to a
        directory that contains a subdirectory with the count data tables for that location (you can specify
        subdirectory='' if the dictionary contains paths directly to the count data). The corresponding tables
        from different locations will be merged together (concatenatd) with a 'location' column added to indicate
        the source of each table row. Each location must have the same data tables, and tables with the same name
        must have the same format.

        Example usage:
        # Assuming the count data for 'Location' is in '/path/to/location/output/count_data'
        locations_paths = {'Ethiopa': '/path/to/ethiopia/output', 'India': '/path/to/india/output'}
        data = VivariumTransformedOutput.merged_from_locations_paths(locations_paths, subdirectory='count_data')
        data.ylls # Merged YLL table for Ethiopia and India
        data.person_time # Merged person-time table for Ethiopia and India
        """
        return cls(load_and_merge_location_count_data(locations_paths, subdirectory))

    def table_names(self):
        return list(self.keys())


def load_transformed_count_data(directory: str, ext=".hdf", **kwargs) -> dict:
    """
    Loads each transformed "count space" .hdf output file into a dataframe,
    and returns a dictionary whose keys are the file names and values are
    are the corresponding dataframes.
    """
    pandas_read = getattr(pd, f"read_{ext[1:]}")
    dfs = {}
    for entry in os.scandir(directory):
        filename_root, extension = os.path.splitext(entry.name)
        if extension == ext:
            #             print(filename_root, type(filename_root), extension, entry.path)
            dfs[filename_root] = pandas_read(entry.path, **kwargs)
    return dfs


def load_count_data_by_location(locations_paths: dict, subdirectory="count_data") -> dict:
    """
    Loads data from all locations into a dictionary of dictionaries of dataframes,
    indexed by location. Each dictionary in the outer dictionary is
    indexed by filename

    For each location, reads data files from a directory called f'{path}/{subdirectory}/',
    where `path` is the path for the location specified in the `locations_paths` dictionary.
    """
    locations_count_data = {
        location: load_transformed_count_data(os.path.join(path, subdirectory))
        for location, path in locations_paths.items()
    }
    return locations_count_data


def merge_location_count_data(locations_count_data: dict, copy=True) -> dict:
    """
    Concatenate the count data tables from all locations into a single dictionary of dataframes
    indexed by table name, with a column added to the begininning of each table specifying the
    location for each row of data.
    """
    if copy:
        # Use a temporary variable and a for loop instead of a dictionary comprehension
        # so we can access the `data` variable later.
        locations_count_data_copy = {}
        for location, data in locations_count_data.items():
            locations_count_data_copy[location] = {
                table_name:
                # Use DataFrame.reindex() to simultaneously make a copy of the dataframe and
                # assign a new location column at the beginning.
                table.reindex(columns=["location", *table.columns], fill_value=location)
                for table_name, table in data.items()
            }
        locations_count_data = locations_count_data_copy
    #         # Alternate version using dictionary comprehension; this version would require a different
    #         # method of iterating through the table names below, because `data` is inaccessible afterwards.
    #         locations_count_data = {
    #             location: {
    #                 table_name:
    #                 table.reindex(columns=['location', *table.columns], fill_value=location)
    #                 for table_name, table in  data.items()
    #             }
    #             for location, data in locations_count_data.items()
    #         }
    else:
        # Modify the dictionaries and dataframes in place
        for location, data in locations_count_data.items():
            for table in data.values():
                # Insert a 'location' column at the beginning of each table
                table.insert(0, "location", location)

    # `data` now refers to the dictionary of count_data tables for the last location
    # encountred in the above for loop. We will use the keys stored in this dictionary
    # to iterate through all the table names and concatenate the tables across all locations.
    all_data = {
        table_name: pd.concat(
            [locations_count_data[location][table_name] for location in locations_count_data],
            copy=False,
            sort=False,
        )
        for table_name in data
    }
    return all_data


def load_and_merge_location_count_data(
    locations_paths: dict, subdirectory="count_data"
) -> dict:
    locations_count_data = load_count_data_by_location(locations_paths, subdirectory)
    return merge_location_count_data(locations_count_data, copy=False)
