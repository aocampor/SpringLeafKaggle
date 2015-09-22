import os, sys

if __name__ == "__main__":
    file1 = open("Data/train.csv", "r")
    i = 0 
    for line in file1:
        lintok = line.rsplit(',')
        print line
        i = i + 1 
        if( i == 10000):
            break

