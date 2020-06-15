import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
import os,glob
N=input('size: ')

col_name=[str(x) for x in range(N)];

# path=os.listdir('./data/');
path=glob.glob('./sol/*.dat');
fig,ax=plt.subplots(ncols=1,nrows=1)
ims=[]

for i in range(1,len(path)+1):
    df='./sol/solution_'+str(i)+'.dat'
    print(df)
    dada=[]
    c = pd.read_csv(df,header=2,sep=" ",names=col_name)
    e=c.reset_index()
    s=N*2;
    d=e.iloc[s:s+N,0:N]
    data=d.values

    ttl = plt.text(0.5, 1.01, 'iter= '+str(i), horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)
    im=plt.imshow(data,cmap='jet',vmin=-0.15, vmax=0, aspect='equal') #vmin vmax

    ims.append([im,ttl])



ani=animation.ArtistAnimation(fig,ims,interval=100,repeat_delay=2000)
ani.save('./figure/solution.gif', writer='imagemagick')

fig2,ax2=plt.subplots(ncols=1,nrows=1)
s=N*3;
d=e.iloc[s:s+N,0:N]
data=d.values
cim=ax2.imshow(data,cmap='jet' ,vmin=-0.15, vmax=0,aspect='equal');
ax2.set_title('analytical solution');
fig2.colorbar(cim)
fig2.savefig('./figure/analytical_sol.png');

fig3,ax3=plt.subplots(ncols=1,nrows=1)
s=N*4;
d=e.iloc[s:s+N,0:N]
data=d.values
cim3=ax3.imshow(np.fliplr(data),cmap='jet', aspect='equal');
ax3.set_title('error');
fig3.colorbar(cim3)

fig3.savefig('./figure/error.png');
