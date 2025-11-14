"""Code used to load files for the CSU Alzheimer's project.
"""
import pandas as pd
from vivarium import Artifact

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
