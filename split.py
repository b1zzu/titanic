#!/usr/bin/env python

# Split the original_train.csv in three parts
# - train   Used to train the model
# - test    Used to test the model
# - prove   Used ony after tests gives nice output


import pandas
from sklearn.utils import shuffle

ORIGINAL_TRAIN = "./data/original_train.csv"

TRAIN = "./data/train.csv"
TEST = "./data/test.csv"
PROVE = "./data/prove.csv"

df = pandas.read_csv(ORIGINAL_TRAIN)

# random shuffle the data
df = shuffle(df)

print('\n## Head:\n')
print(df.head())

length = len(df)
train_length = int(length * 0.8)
test_length = int(length * 0.1)
prove_length = int(length * 0.1)

# fix prove length
prove_length -= ((train_length + test_length + prove_length) - length)

assert (prove_length + test_length + train_length) == length

print('\n## Lengths:\n')
print('Total: {}'.format(length))
print('Train: {}'.format(train_length))
print('Test:  {}'.format(test_length))
print('Prove: {}'.format(prove_length))

train = df[:train_length]
test = df[train_length:train_length+test_length]
prove = df[train_length+test_length:]

assert len(train) == train_length
assert len(test) == test_length
assert len(prove) == prove_length

train.to_csv(TRAIN)
test.to_csv(TEST)
prove.to_csv(PROVE)
