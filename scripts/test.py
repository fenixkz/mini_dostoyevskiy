from data import get_dataset


train, val = get_dataset()

print(val[:1000])