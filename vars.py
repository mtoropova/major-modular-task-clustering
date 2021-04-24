options = ['shots', 'dribbles', 'passes', 'tackles', 'interceptions', 'fouls_committed', 'shots_conceded']

general_info_columns = ['teamName', 'teamRegionName']

columns_in_options = {
    'shots': ['shotsTotal'],
    'tackles': ['tackleTotalAttempted'],
    'passes': ['passTotal'],
    'dribbles': ['dribbleTotal'],
    'fouls_given': ['foulGiven'],
    'fouls_committed': ['foulCommitted'],
    'interceptions': ['interceptionAll'],
    'key_passes': ['keyPassesTotal'],
    'possession_loss': ['dispossessed'],
    'clearances': ['clearanceTotal'],
    'saves': ['saveTotal'],
    'shots_conceded': ['shotsConcededPerGame'],
    'shots_counter': ['shotCounter'],
    'shots_open': ['shotOpenPlay']
}

norm_names_options = {
    'shots': 'удары',
    'dribbles': 'дриблинг',
    'passes': 'пасы',
    'tackles': 'отборы',
    'interceptions': 'перехваты',
    'fouls_committed': 'фолы',
    'shots_conceded': 'пропущенные удары',
}

dend_names = {'inital': 'inital_data', 'norm': 'norm_data', 'stand': 'stand_data'}

data_states = ['inital', 'norm', 'stand']
