import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
#outdir='Results/000038'
#eigen_pratiofile=outdir + '/' + 'eigen_pratio.csv'
epfile_log = 'pratio_eigen.png'
eigenvalues_list=[]
pratio_list=[]
columns = defaultdict(list) # each value in each column is appended to a list
#folder=['000039','000033','000034','000035','000036']
folder=['000039']
for i in range(len(folder)):
    columns = defaultdict(list)
    outdir='Results/'+folder[i]
    eigen_pratiofile=outdir + '/' + 'eigen_pratio.csv'
    with open(eigen_pratiofile) as f:
       fieldnames = ['eigenvalue','participation_ratio']
       reader = csv.DictReader(f) # read rows into a dictionary format
       for row in reader: # read a row as {column1: value1, column2: value2,...}
           for (k,v) in row.items(): # go over each column name and value 
              columns[k].append(np.float128(v)) # append the value into the appropriate list
    eigenvalues_list+=np.array(columns['eigenvalue']).tolist()
    pratio_list+=np.array(columns['participation_ratio']).tolist()

plt.clf()
plt.yscale('log')
plt.ylim(0.003,0.06)
plt.yticks([0.003,0.01,0.06],['$3\cdot 10^{-3}$','$10^{-2}$','$6\cdot 10^{-2}$'])
plt.scatter(eigenvalues_list,pratio_list,s=1)
plt.xlabel('eigenvalues')
plt.ylabel('p-ratio')
plt.savefig(epfile_log,dpi=300)
print("> pratio vs eigenvalues to {}".format(epfile_log))

