import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 500)
# Define data locations
TITANIC_TRAINING = 'data/train.csv'
TITANIC_TEST = 'data/test.csv'
train_set = pd.read_csv(TITANIC_TRAINING)
# No lines without header
train_set_no_lines = sum(1 for row in open(TITANIC_TRAINING)) - 1
test_set = pd.read_csv(TITANIC_TEST)


def echo_missing_values(dataset):
    print(dataset.columns[dataset.isnull().any()])


# We combine the training and testing set to have a more complete set when creating our
# new features
def get_combined_data():
    # We do not want the labels (survived or not) when merging se we drop this column
    tmp = train_set.drop(['Survived'], 1)
    # Combining the sets
    combined = tmp.append(test_set)
    # The passengerID is irrelevant and thus we can drop it.
    combined.drop(['PassengerId'], 1, inplace=True)

    return combined


# The names can be useful as they contain titles. From history we know that the richest
# passengers were more likely to survive, and thus we can attempt to make sense of the
# passengers social status by their titles and use this as a feature.
def add_social_status(dataset):
    # We can classify each title to a given status
    # (We'll rank this from 0-2)
    # The title dictionary is extracted from the test set
    title_dic = {
        "Lady": 2,
        "Jonkheer": 2,
        "Don": 2,
        "Sir": 2,
        "Rev": 2,
        "the Countess": 2,
        "Dr": 1,
        "Capt": 1,
        "Col": 1,
        "Major": 1,
        "Master": 1,
        "Mme": 0,
        "Mlle": 0,
        "Ms": 0,
        "Mr": 0,
        "Mrs": 0,
        "Miss": 0
    }
    # Then we will add it to the returned and improved set
    # First we add the new column to the dataset
    dataset['social_status'] = None
    for index, row in dataset.iterrows():
        title = row['Name'].split(',')[1].split('.')[0].strip()
        try:
            social_status = title_dic[title]
        except KeyError:
            # If we should encounter a title we do not know about, we assume a common status
            social_status = 0
        dataset.loc[index, 'social_status'] = social_status

    return dataset


# Since many rows are missing age we must clean this up.
# We can be smarter than just filling in the average value, by looking at some other rows.
# The titles from the names, gender and the class of the travel are good candidates for this. Older people
# Are more likely to travel higher classes.
def add_ages(dataset):
    overall_average = dataset['Age'].mean().round(0)
    # We first extract the train_set with social status
    grouped_average = dataset.iloc[:train_set_no_lines].groupby(['Sex', 'Pclass', 'social_status'])
    with_age_mean = grouped_average.mean().round(0)
    with_age = with_age_mean.reset_index()
    age_decidor = with_age[['Sex', 'Pclass', 'social_status', 'Age']]
    # Now that we have our table with more accurate average ages we can use that to fill in the blanks.
    for index, row in dataset.iterrows():
        if np.isnan(row['Age']):
            try:
                # We first define our condition for a match in our age decidor.
                row['social_status'] = 5
                condition = (
                        (age_decidor['Sex'] == row['Sex']) &
                        (age_decidor['social_status'] == row['social_status']) &
                        (age_decidor['Pclass'] == row['Pclass'])
                )
                dataset.loc[index, 'Age'] = age_decidor[condition]['Age'].values[0]
            except IndexError:
                # If we encounter an unknown key in the test set we will assume the overall
                # average age.
                dataset.loc[index, 'Age'] = overall_average
    return dataset


def remove_useless_columns(dataset):
    dataset.drop('Name', axis=1, inplace=True)
    dataset.drop('Embarked', axis=1, inplace=True)
    dataset.drop('Pclass', axis=1, inplace=True)
    dataset.drop('Ticket', axis=1, inplace=True)
    return dataset


# We fill the missing values in the embarked col with the most common from the training set
def add_embarked(dataset):
    most_frequent = dataset.iloc[:train_set_no_lines]['Embarked'].mode()[0]
    dataset['Embarked'].fillna(most_frequent, inplace=True)
    # Now we add dummy encoding. The Embarked column is later dropped in the remove_useless_columns func
    dummies = pd.get_dummies(dataset['Embarked'], prefix='Embarked')
    dataset = pd.concat([dataset, dummies], axis=1)
    return dataset


# We add the missing values in the fares col with the average
def add_fares(dataset):
    avg = dataset.iloc[:train_set_no_lines]['Fare'].mean().round(0)
    dataset['Fare'].fillna(avg, inplace=True)
    return dataset


# We clean the cabins to the dorm letter and add an U for unknown
def clean_cabins(dataset):
    dataset['Cabin'].fillna('U', inplace=True)
    # We now assign a numeric value based on the survivability in each dorm from the testing set
    for index, row in dataset.iterrows():
        dataset.loc[index, 'Cabin'] = row['Cabin'][0]
    return dataset


# What cabin dorms are most likely to survive? We will later arrange the cabins from 0-n based
# on the survivability
def get_survivability_by_dorm_dic():
    # We get our data from the training set
    tmp = pd.read_csv(TITANIC_TRAINING)
    tmp = clean_cabins(tmp)
    # We are only intrested in survivability by cabin
    tmp = tmp[['Cabin', 'Survived']]
    surv_counts = tmp.groupby(['Cabin']).sum()
    total = tmp['Cabin'].value_counts()
    total_counts = pd.DataFrame({'Cabin': total.index, 'total_count': total.values})
    # We merge the survived and total counts to obtain the percentage of survivability
    merged = pd.merge(surv_counts, total_counts, on='Cabin')
    merged['percentage_survivors'] = merged['Survived'] / merged['total_count']
    # Then we sort and reset index so we can later iterate in the order of survivability
    merged.sort_values(by=['percentage_survivors'], inplace=True, ascending=True)
    merged.reset_index(drop=True, inplace=True)
    # Now we'll crate a dic assigning a survivability score from 0-n based on the survivability
    # on the dorm
    surv_by_dorm = dict()
    for index, row in merged.iterrows():
        surv_by_dorm[row['Cabin']] = index

    return surv_by_dorm


def normalize_cabins(dataset):
    surv_by_dorm = get_survivability_by_dorm_dic()
    for index, row in dataset.iterrows():
        dataset.loc[index, 'Cabin'] = surv_by_dorm[row['Cabin'][0]]
    return dataset


def normalize_gender(dataset):
    dataset['Sex'] = dataset['Sex'].map({'male': 1, 'female': 0})
    return dataset


def dummy_encode_pclass(dataset):
    dummies = pd.get_dummies(dataset['Pclass'], prefix='Pclass')
    dataset = pd.concat([dataset, dummies], axis=1)
    return dataset


# Extracts the ticket type from the full ticket
def getTicketType(ticket):
    ticket = ticket.replace('.', '')
    ticket = ticket.replace('/', '')
    ticket = ticket.split()
    ticket = ticket[0].strip()
    return ticket


def dummy_encode_tickets_by_prefix(dataset):
    for index, row in dataset.iterrows():
        if row['Ticket'].isnumeric():
            dataset.loc[index, 'Ticket'] = 'Norm'
        else:
            dataset.loc[index, 'Ticket'] = getTicketType(row['Ticket'])
    tickets_dummies = pd.get_dummies(dataset['Ticket'], prefix='Ticket')
    return pd.concat([dataset, tickets_dummies], axis=1)


# print_missing_values(train_set)

# This now returns an empty array :)
# echo_missing_values(data)


def get_train_test_targets():
    log('Loading Data')
    data = get_combined_data()
    log('Done')
    log('Processing Social Status')
    data = add_social_status(data)
    log('Done')
    log('Processing Ages')
    data = add_ages(data)
    log('Done')
    log('Processing Embarked')
    data = add_embarked(data)
    log('Done')
    log('Processing Fares')
    data = add_fares(data)
    log('Done')
    log('Processing Cabins')
    data = clean_cabins(data)
    data = normalize_cabins(data)
    log('Done')
    log('Processing Genders')
    data = normalize_gender(data)
    log('Done')
    log('Dummy Encoding Pclass & Tickets')
    data = dummy_encode_pclass(data)
    data = dummy_encode_tickets_by_prefix(data)
    log('Done')
    log('Dropping unwanted Columns')
    data = remove_useless_columns(data)
    log('Done')
    return data.iloc[:train_set_no_lines], data[train_set_no_lines:], \
           pd.read_csv('data/train.csv')['Survived'].values


def log(string):
    debug = False
    if debug:
        print(string)
