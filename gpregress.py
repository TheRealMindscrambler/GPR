
import numpy as np
from matplotlib import pyplot as plt
import GPy
from numpy import linalg as LA

with open('E1.txt') as f:
    E1s = []
    for line in f:
        line = line.split() 
        if line:            
            line = [float(i) for i in line]
            E1s.append(line)

with open('E2.txt') as f:
    E2s = []
    for line in f:
        line = line.split()  
        if line:            
            line = [float(i) for i in line]
            E2s.append(line)

with open('E3.txt') as f:
    E3s = []
    for line in f:
        line = line.split()  
        if line:            
            line = [float(i) for i in line]
            E3s.append(line)

with open('Q1.txt') as f:
    Q1s = []
    for line in f:
        line = line.split() # to deal with blank 
        if line:            # lines (ie skip them)
            line = [float(i) for i in line]
            Q1s.append(line)

E1=np.matrix(E1s)
E2=np.matrix(E2s)
E3=np.matrix(E3s)
Q1=np.matrix(Q1s)

n=1
E1new = []
E2new = []
E3new = []
Q1new = []
E1save = []
E2save = []
E3save = []
Q1save = []
for x in range(0,1499):
    if Q1s[x][3]>0:
        if n<10:
            line=Q1s[x][0:3]
            Q1new.append(line)
            E1new.append(E1s[x])
            E2new.append(E2s[x])
            E3new.append(E3s[x])
            n=n+1
        else:
            line=Q1s[x][0:3]
            Q1save.append(line)
            E1save.append(E1s[x])
            E2save.append(E2s[x])
            E3save.append(E3s[x])
            n=1
    

Q1reduced=np.matrix(Q1new)
E1reduced=np.matrix(E1new)
E2reduced=np.matrix(E2new)
E3reduced=np.matrix(E3new)
Q1side=np.matrix(Q1save)
E1side=np.matrix(E1save)
E2side=np.matrix(E2save)
E3side=np.matrix(E3save)


Q1n = []
for x in range(0,len(Q1reduced)):
    line=LA.norm(Q1reduced[x,:])
    Q1n.append(line)

Q1ns = []
for x in range(0,len(Q1side)):
    line=LA.norm(Q1side[x,:])
    Q1ns.append(line)

Q1norm=np.matrix(Q1n)
Q1norm=np.transpose(Q1norm)

Q1norms=np.matrix(Q1ns)
Q1norms=np.transpose(Q1norms)

E13=E1reduced-E3reduced
E23=E2reduced-E3reduced

E13s=E1side-E3side
E23s=E2side-E3side

kernel = GPy.kern.RBF(input_dim=3)

m = GPy.models.GPRegression(Q1reduced,E13,kernel)
# m.optimize()

print(m)

# fig = m.plot(plot_density=True)
# plt.show()


Ep= m.predict(Q1side)[0]

















