import zipfile
import pandas as pd
import numpy as np

''' All functions return two dataframes, one of the nodes and one of the edges '''

''' Function that loads one of the AddHealth schools.
It takes as input the number of the dataset in AddHealth and the threshold of the proportion for one category to be considered '''

def load_AddHealth(NUM_SCHOOL, th_sex = 0.3, th_race=0.1, th_grade=0.05, school=0, remove_unreported=False,
                  data_path = r"./Datasets/AddHealth"):

  # load the school given by NUM_SCHOOL
  with zipfile.ZipFile(data_path+"/comm"+str(NUM_SCHOOL)+".csv.zip") as data_zip:
    with data_zip.open('edges.csv') as myfile:
      edges_list = pd.read_csv(myfile, dtype='category')
    with data_zip.open('nodes.csv') as myfile:
      nodes_list = pd.read_csv(myfile, dtype='category')
  nodes_list = nodes_list.rename(columns={'# index':'index',' sex':'sex', ' race':'race', ' grade':'grade', ' school':'school'})
  edges_list = edges_list.rename(columns={'# source':'source',' target':'target', ' activities':'activities'})

  # replace the numbers by the categories
  sex = {'0':'Unreported','1':'Male','2':'Female'}
  race = {'0':'Unreported','1':'White','2':'Black','3':'Hispanic','4':'Asian','5':'Mixed/other'}
  grade = {'0':'Unreported','6':'6th','7':'7th','8':'8th','9':'9th','10':'10th','11':'11th','12':'12th'}
  nodes_list.replace({'sex': sex,'race':race,'grade':grade},inplace=True)
  nodes_list.drop([' _pos'],axis=1,inplace=True)
    
  # remove unreported
  if remove_unreported:
    nodes_list.drop(nodes_list[(nodes_list == "Unreported").any(axis=1)].index, inplace=True)

 # save the counts of each category in the data
  n_sex = nodes_list['sex'].value_counts()
  n_race = nodes_list['race'].value_counts()
  n_grade = nodes_list['grade'].value_counts()

  # save the proportion of each category in the data
  prop_sex = n_sex.div(int(n_sex.sum(axis=0)))
  prop_race = n_race.div(int(n_race.sum(axis=0)))
  prop_grade = n_grade.div(int(n_grade.sum(axis=0)))

  # only select the categories that are above the selected thresholds (if <1: proportion, if >=1: abs. value)
  if th_sex<1: nodes_list = nodes_list[(nodes_list['sex'].isin(prop_sex.index[np.where(prop_sex>th_sex)[0]]))]
  else: nodes_list = nodes_list[(nodes_list['sex'].isin(n_sex.index[np.where(n_sex>th_sex)[0]]))]
  if th_race<1: nodes_list = nodes_list[(nodes_list['race'].isin(prop_race.index[np.where(prop_race>th_race)[0]]))]
  else: nodes_list = nodes_list[(nodes_list['race'].isin(n_race.index[np.where(n_race>th_race)[0]]))]
  if th_grade<1: nodes_list = nodes_list[(nodes_list['grade'].isin(prop_grade.index[np.where(prop_grade>th_grade)[0]]))]
  else: nodes_list = nodes_list[(nodes_list['grade'].isin(n_grade.index[np.where(n_grade>th_grade)[0]]))]

  # if there is only one school select that, otherwise select the one given by the 'school' variable. If None, select all
  if 'school' in nodes_list:
    num_schools = len(nodes_list['school'].unique())
    print(f'Community #{NUM_SCHOOL}: {num_schools} schools')
    if school is not None: 
      nodes_list = nodes_list[(nodes_list['school'] == str(school))]
  else: print(f'Community #{NUM_SCHOOL}: one school')

  # select the edges relative to the selected nodes
  edges_list = edges_list[(edges_list['source'].isin(nodes_list['index'].values)) & (edges_list['target'].isin(nodes_list['index'].values))]

  # remove the unused categories and add information to the edges_list
  nodes_list = nodes_list.set_index('index')
  for category in ['sex','race','grade']:
    nodes_list[category] = nodes_list[category].cat.remove_unused_categories()
    edges_list['source '+category] =  edges_list['source'].map(nodes_list[category])
    edges_list['target '+category] =  edges_list['target'].map(nodes_list[category])

  return nodes_list,edges_list

def build_nodes_edges_input_df(nodes_df, edges_df, dimensions):

  edges_df = edges_df.copy()
  nodes_df = nodes_df.copy()

  for dim in dimensions:
      nodes_df[dim] = nodes_df[dim].cat.remove_unused_categories()
      edges_df['source '+dim] =  edges_df['source'].map(nodes_df[dim])
      edges_df['target '+dim] =  edges_df['target'].map(nodes_df[dim])

  return nodes_df, edges_df

''' Function that loads one of the datasets of the University in Bogota '''

def load_Bogota(department, wave=1):
   
    nodes_list = pd.read_csv(f'./Datasets/Bogota data/Networks_SPP_in_english/Nodes_{department}.csv', sep='|', dtype='category')
    edges_list = pd.read_csv(f'./Datasets/Bogota data/Networks_SPP_in_english/Edges_Friendship_network_directed_{department}_{2016+wave}2.csv', sep='|', dtype='category')

    SES_dict = {'0':'Poor','1':'Poor','2':'Poor','3':'Poor','4':'Rich','5':'Rich','6':'Rich'}
    CITIES_dict = {'BOGOTA D.C.':'Bogota', 'IBAGUE':'Colombia out', 'COTA':'Outskirts', 'BARRANCABERMEJA': 'Colombia out', 'NEIVA':'Colombia out', 'LA CALERA':'Outskirts', 'CHIA':'Outskirts', 'LA DORADA':'Colombia out', 'CUCUTA':'Colombia out','RETIRO':'Colombia out','DUITAMA':'Colombia out','BUGA':'Colombia out', 'BUCARAMANGA':'Colombia out', 'PAMPLONA':'Colombia out','CARMEN DE VIBORAL':'Colombia out','ARMENIA':'Colombia out','Unknown':'Unknown'}

    nodes_list['Place_of_born'] = nodes_list['Place_of_born'].cat.add_categories('Unknown')
    nodes_list['Place_of_born'] = nodes_list['Place_of_born'].fillna('Unknown')

    nodes_list = nodes_list.replace({'Place_of_born':CITIES_dict,'SES':SES_dict,'GPA_2017':{'0.0':'2.5'},'GPA_2018':{'0.0':'2.5'}})
    nodes_list['GPA_2017'] = nodes_list['GPA_2017'].fillna('2.5')
    nodes_list['GPA_2018'] = nodes_list['GPA_2018'].fillna('2.5')

    return nodes_list, edges_list



''' Function that loads a specific wave of the school in Glasgow '''

def load_Glasgow(wave=1):
   
    nodes_list = pd.read_csv('./Datasets/Glasgow_data/Nodes_list.csv', dtype='category')
    edges_mat = pd.read_csv(f'./Datasets/Glasgow_data/Edges_list_t{wave}.csv', dtype='category')

    N_NODES = nodes_list.shape[0]
    not_na_t1 = nodes_list['Tobacco-t1'].notna()
    indexes = [i for i in range(N_NODES) if not_na_t1[i]==True]
    edges_list = []
    for i in indexes:
        for j in indexes:
            if edges_mat.iloc[i,j]=='1': edges_list.append((i,j,{'Strength':'Best friend'}))
            if edges_mat.iloc[i,j]=='2': edges_list.append((i,j,{'Strength':'Just a friend'}))

    TOBACCO_dict = {'Non':'No','Occasional':'Yes', 'Regular':'Yes'}
    CANNABIS_dict = {'Non':'No','Tried once':'No','Occasional':'Yes', 'Regular':'Yes'}
    nodes_list.replace({'Tobacco-t1': TOBACCO_dict, 'Cannabis-t1': CANNABIS_dict},inplace=True)

    edges_list = pd.DataFrame(edges_list, columns=['source','target','strength'])

    return nodes_list, edges_list

