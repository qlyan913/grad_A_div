import numpy as np
import matplotlib.pyplot as plt
id=16
filename="Results/{:06d}/eigenvalues.txt".format(id)
plot_file="Results/{:06d}/es_hist.png".format(id)
plot_file2="Results/{:06d}/es_r_hist.png".format(id)
eigens=np.loadtxt(filename)
es = [eigens[i+1]-eigens[i] for i in range(len(eigens)-1)]
es = np.array(es)

r = [min(es[i],es[i+1])/max(es[i],es[i+1]) for i in range(len(es)-1)]
average=sum(es)/len(es)
es=es/average  # normalize
x=np.linspace(0,8,1000)
y=np.exp(-x)
plt.clf()
plt.hist(es,bins=60,density=True)
plt.plot(x,y,"r-")
plt.xlabel('s')
plt.title('energy spacing with average energy space {}'.format(average))
plt.savefig(plot_file)
print("> histogram plotted to {}".format(plot_file))

average2=sum(r)/len(r)
plt.clf()
plt.hist(r,bins=60,density=True)
x=np.linspace(0,1,100)
y=2/(1+x)**2
plt.plot(x,y,"r-")
plt.xlabel('s')
plt.title('energy spacing with average energy space ratio {}'.format(average2))
plt.savefig(plot_file2)
print("> histogram plotted to {}".format(plot_file2))
