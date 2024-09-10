import numpy as np
import matplotlib.pyplot as plt
id=19
filename="Results/{:06d}/eigenvalues.txt".format(id)
plot_file="Results/{:06d}/es_hist.png".format(id)
eigens=np.loadtxt(filename)
es = [eigens[i+1]-eigens[i] for i in range(len(eigens)-1)]
es = np.array(es)

average=sum(es)/len(es)
es=es/average  # normalize
x=np.linspace(0,8,1000)
y=np.exp(-x)
plt.clf()
plt.hist(es,bins=60,weights=np.ones(len(es))/len(es))
plt.plot(x,y,"r-")
plt.xlabel('s')
plt.title('energy spacing with average energy space {}'.format(average))
plt.savefig(plot_file)
print("> histogram plotted to {}".format(plot_file))
