import pickle

with open('trained_cmac.pkl', 'rb') as f:
    cmac_test = pickle.load(f)
print(cmac_test)