import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    traindf = pd.read_csv('../Data/train.csv', error_bad_lines = False , index_col=False, dtype='unicode')
    
    columns =  traindf.columns  
    #print columns
    for item in columns:
        frequencies = {}
        if(item != 'ID'):
            for it in traindf[item]:
                if it in frequencies:
                    frequencies[it] = frequencies[it] + 1
                else:
                    frequencies[it] = 1
            plt.bar(range(len(frequencies)), frequencies.values(), align="center")
            plt.xticks(range(len(frequencies)), list(frequencies.keys()))
            plt.yscale('log')
            plt.savefig('../Fig/' + item + '.png')       