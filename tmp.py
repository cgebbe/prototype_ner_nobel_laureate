from datasets import Dataset

my_dict = {'id': [0, 1, 2],
           'name': ['mary', 'bob', 'eve'],
           'age': [24, 53, 19]}

ds = Dataset.from_dict(my_dict)