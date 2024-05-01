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

    def get_person_time_table_name(self, colnames, exclude=None):
        """Return the name of a person-table that contains the
        specified columns, None if none can be found.
        """
        person_time_table_name = next(
            # Set default to None if no tables found
            find_person_time_tables(self, colnames, exclude), None)
        return person_time_table_name
