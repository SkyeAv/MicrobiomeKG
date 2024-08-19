# v0.8 08/13/24 Skye Goetz (ISB)

import os
import re
import sys
import copy
import gzip
import yaml
import dill
import subprocess
import pandas as pd

def load_pickle(file_path):
    with gzip.open(file_path, 'rb') as file:
        return dill.load(file)

def initialize_paths(cwd, config_path):
    results_path = os.path.join(cwd, 'results', f'{os.path.splitext(os.path.basename(config_path))[0]}.tsv')
    pickle_path = os.path.join(cwd, 'pickle')
    return results_path, pickle_path

def load_data(pickle_path, cwd):
    files = ['name_resolver', 'term_mapper', 'node_normalizer', 'semantic_types']
    try:
        return {file: load_pickle(os.path.join(pickle_path, f'{file}.pkl.gz')) for file in files}
    except FileNotFoundError:
        subprocess.run(['python3', os.path.join(cwd, 'bin', 'BABEL2pickle.py')], check=True)
        return {file: load_pickle(os.path.join(pickle_path, f'{file}.pkl.gz')) for file in files}

def cutoff(df, config):
    column = config['cutoff']['column']
    mode = config['cutoff']['mode']
    try: value = float(config['cutoff']['value'])
    except ValueError: value = str(config['cutoff']['value'])
    if 'greater_than_or_equal_to' in mode:
        df = df[df[column].astype(float) >= value]
    if 'less_than_or_equal_to' in mode:
        df = df[df[column].astype(float) <= value]
    if 'if_equals' in mode:
        df = df[df[column].astype(str) != str(value)]
    return df

def format_column(df, column, config):
    column_config = config.get(column, {})
    if 'explode' in column_config:
        df[column] = df[column].str.split(column_config['explode']['delimiter']).explode().reset_index(drop=True)
    if 'prefix' in column_config:
        for prefix in column_config['prefix']:
            df[column] = prefix['prefix'] + df[column].astype(str)
    if 'text_replacements' in column_config:
        for replacement in config[column]['text_replacements']:
            if replacement['replacement'] == None: df[column] = df[column].astype(str).apply(lambda x: re.sub(replacement['pattern'], '', x))
            else: df[column] = df[column].astype(str).apply(lambda x: re.sub(replacement['pattern'], replacement['replacement'], x))
    if 'curie' not in config[column] and 'curie_column_name' not in config[column]:
        df[column] = df[column].str.lower()
    return df

def map_column(df, column, config, mappings):
    node_normalizer = mappings['node_normalizer']
    name_resolver = mappings['name_resolver']
    term_mapper = mappings['term_mapper']
    name_column = f'{column}_name'
    if 'curie' in config[column]:
        df[column] = config[column]['curie']
        df = format_column(df, column, config)
        df[column] = df[column].map(node_normalizer).fillna(df[column])
        df[name_column] = df[column].map(name_resolver).fillna(df[column])
    elif 'curie_column_name' in config[column]:
        df = df.rename(columns={config[column]['curie_column_name']: column})
        df = format_column(df, column, config)
        df[column] = df[column].map(node_normalizer).fillna(df[column])
        df[name_column] = df[column].map(name_resolver).fillna(df[column])
    elif 'value' in config[column]:
        df[column] = config[column]['value']
        df = format_column(df, column, config)
        df[column] = df[column].map(term_mapper).fillna('nan')
        df = df[df[column] != 'nan']
        df[column] = df[column].map(node_normalizer).fillna(df[column])
        df[name_column] = df[column].map(name_resolver).fillna(df[column])
    elif 'name_column_name' in config[column]:
        df = df.rename(columns={config[column]['name_column_name']: column})
        df = format_column(df, column, config)
        df[column] = df[column].map(term_mapper).fillna('nan')
        df = df[df[column] != 'nan']
        df[column] = df[column].map(node_normalizer).fillna(df[column])
        df[name_column] = df[column].map(name_resolver).fillna(df[column])
    return df

def values(value):
    try:
        value = float(value)
        return format(value, '.2e') if value >= 1e5 else value
    except (ValueError, TypeError):
        return value

def sci_notation(df):
    for column in ['p', 'relationship_strength']:
        if column in df.columns: 
            df[column] = df[column].apply(lambda x: values(x))
    return df

def config2KG(config, mappings, results_path):
    df = pd.read_excel(config['data_location']['path_to_xlsx'], 
        sheet_name=config['data_location']['sheet_to_use'], 
        header=config['data_location']['header'])
    if 'last_line' in config['data_location']:
        df = df.iloc[:config['data_location']['last_line']]
    for provenance, value in config['provenance'].items():
        df.insert(0, provenance, value)
    df.insert(0, 'predicate', config['predicate']['value'])
    for attribute, attr_config in config['attributes'].items():
        if attribute in df.columns:
            if attr_config.get('column_name') != attribute:
                df.drop(columns=[attribute], inplace=True)
        if 'value' in attr_config:
            df.insert(0, attribute, attr_config['value'])
        if 'column_name' in attr_config:
            df.rename(columns={attr_config['column_name']: attribute}, inplace=True)
    if 'cutoff' in config:
        df = cutoff(df, config)
    for column in ['subject', 'object']:
        df = map_column(df, column, config, mappings)
    df = df[['subject', 'predicate', 'object', 'subject_name', 'object_name', 'n', 'relationship_strength', 'p', 'relationship_type', 
        'p_correction_method', 'knowledge_level', 'agent_type', 'publication', 'publication_name', 'authors', 'url', 'sheet_name', 'curator_name']]
    df.dropna(inplace=True)
    df.drop_duplicates(subset=['subject', 'predicate', 'object', 'publication', 'relationship_type'], keep='first', inplace=True)
    semantic_types = mappings['semantic_types']
    for column in ['object', 'subject']:
        category_col = f'{column}_category'
        df.insert(15, category_col, (df[column].map(semantic_types).fillna('nan')))
        df = df[df[category_col] != 'nan']
    if not os.path.isfile(results_path): sci_notation(df).to_csv(results_path, sep='\t', index=False)
    else:
        df = pd.concat([pd.read_csv(results_path, sep='\t'), pd.DataFrame(df)], ignore_index=True)\
        .pipe(sci_notation).to_csv(results_path, sep='\t', index=False)

def main():
    cwd = os.getcwd()
    os.makedirs(f'{cwd}/results', exist_ok=True)
    try:
        config_path = sys.argv[1]
    except IndexError:
        print('config2KG : Cannot Access Necessary Files')
        return
    results_path, pickle_path = initialize_paths(cwd, config_path)
    if os.path.isfile(results_path):
        return
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    mappings = load_data(pickle_path, cwd)
    with open(config_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    if 'sections' in config:
        for section in config['sections']:
            temp_config = copy.deepcopy(config)
            for subsection in section:
                if subsection in temp_config:
                    temp_config[subsection].update(section[subsection])
            config2KG(temp_config, mappings, results_path)
    else:
        config2KG(config, mappings, results_path)

if __name__ == '__main__':
    main()
