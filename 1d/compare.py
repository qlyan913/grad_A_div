import numpy as np
import matplotlib.pyplot as plt
id=25
filename="Results/{:06d}/eigenvalues.txt".format(id)
eigen_b=np.loadtxt(filename)

id=24
filename="Results/{:06d}/eigenvalues.txt".format(id)
eigen_10L_deg1=np.loadtxt(filename)
id=26
filename="Results/{:06d}/eigenvalues.txt".format(id)
eigen_L_deg1=np.loadtxt(filename)
id=27
filename="Results/{:06d}/eigenvalues.txt".format(id)
eigen_L_deg5=np.loadtxt(filename)

diff_10L_deg1=abs(eigen_10L_deg1-eigen_b)
diff_L_deg1=abs(eigen_L_deg1-eigen_b)
diff_L_deg5=abs(eigen_L_deg5-eigen_b)
modes=np.array([d+1 for d in range(len(eigen_b))])
plt.clf()
plt.yscale('log')
plt.scatter(modes,diff_10L_deg1,label='nelts=10*L,deg=1')
plt.scatter(modes,diff_L_deg5,label='nelts=L,deg=5')
plt.scatter(modes,diff_L_deg1,label='nelts=L,deg=1')
plt.legend(fontsize=10)
plt.xlabel('mode')
plt.title('difference to eigenvalues with nelts=10*L,deg=5')
plot_file="eigenvalues_diff.png"
plt.savefig(plot_file)
print("plotted to {}".format(plot_file))

diff_10L_deg1=abs(eigen_10L_deg1-eigen_b)/eigen_b
diff_L_deg1=abs(eigen_L_deg1-eigen_b)/eigen_b
diff_L_deg5=abs(eigen_L_deg5-eigen_b)/eigen_b
modes=np.array([d+1 for d in range(len(eigen_b))])
plt.clf()
plt.yscale('log')
plt.scatter(modes,diff_10L_deg1,label='nelts=10*L,deg=1')
plt.scatter(modes,diff_L_deg5,label='nelts=L,deg=5')
plt.scatter(modes,diff_L_deg1,label='nelts=L,deg=1')
plt.legend(fontsize=10)
plt.xlabel('mode')
plt.title('relative difference to eigenvalues with nelts=10*L,deg=5')
plot_file="eigenvalues_rel_diff.png"
plt.savefig(plot_file)
print("plotted to {}".format(plot_file))

