import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt


def initialstate(N):   
    ''' generates a random spin configuration for initial condition'''
    state =  2*np.random.randint(2, size=(N,N))-1#np.ones((N,N),dtype= int) #
    return state
    
    
def mcmove(config, beta,N,T):
    '''Monte Carlo move using Metropolis algorithm '''
    hist = np.zeros(N*N)
    for i in range(N):
        for j in range(N):
                a = np.random.randint(0, N)
                b = np.random.randint(0, N)
                s =  config[a, b]
                sprim = s
                nb = config[(a+1)%N,b] + config[a,(b+1)%N] + config[(a-1)%N,b] + config[a,(b-1)%N]
                cost = 2*s*nb
                if cost < 0:
                    s *= -1
                elif rand() < np.exp(-cost*beta):
                    s *= -1
                config[a, b] = s
                #if sprim != s:
    if T<2.27:
      dddd = clustering(N,config,np.sum(config))# magna(config,N,np.sum(config))#
    else:
      Een = calcEnergy(config,N)
      dddd =   Label(N, Een)#magna(Een,N)#siz(N,Een)#,config,np.sum(config))#siz(N,Een)# magna(config,N,np.sum(config))#ee
      dddd = Label1(N,dddd)
    out = np.logical_not(dddd == 0)
    dddd = dddd[out]
    unique, counts = np.unique(np.array(dddd), return_counts=True)
    unique, counts = np.unique(counts, return_counts=True)
    for i in unique:
        hist[i-1] += counts[int(np.where(unique == i)[0])]    
    if sum(hist) != 0.0:
        pi = [(hist[k]/np.sum(hist))for k in range(len(hist))]
        ss = [ (k*pi[k])for k in range(len(hist))]
        out = np.logical_not(hist == 0)
        dddd = hist[out]
        #print(dddd)
        Sis = sum(ss)/sum(pi)
        hist_indexx = round(Sis)
        Nun = hist[hist_indexx]#sum(dddd)/len(dddd)
    else:
        Sis = 0.0
        Nun = 0.0
    return config,Sis,Nun 
    
def Find(L, a, b, c, d, Lw): #Busca elementos en matriz.
    x = L[a,b];
    y = L[c,d];
    find_rc = Find_prim(L,x,Lw);
    row = find_rc[0];
    col = find_rc[1];

    for i in range(0,len(col)):
        aa = row[i];
        bb = col[i];
        L[aa,bb] = y; #cambie item
    return L
def Find_prim(a, b, Lw): #Complemento de la funcion Find.
    #size = a.shape;
    row = np.array([], dtype=np.int64);
    col = np.array([], dtype=np.int64);
    #if len(b) == 1:
    for i in range(0,Lw):
        for j in range(0,Lw):
            if a[i,j] == b:
                row = np.append(row, i);
                col = np.append(col, j);
                #elif len(b) > 1: #Posible trabajo futuro.
                    #size_b = b.shape;
                    #for i in range(0,size[0]):
                        #for j in range(0,size[1]):
                            #for k in range(size_b[1]):
                                #if a[i,j] == b[k]:
                                    #row = np.append(row, i);
                                    #col = np.append(row, j); 
    return [[row],[col]]
def Label1(L, R): #Crea la matriz de clusters.
    iD = 1;
    label = np.zeros((L,L));
    for i in range(0,L):
        for j in range(0,L):
            if R[i,j]:
                l_a = Above_left(i,j,R);
                above = l_a[0];
                left = l_a[1];

                if left == 0 and above == 0:
                    label[i,j] = iD;
                    iD = iD + 1;
                elif left != 0 and above == 0:
                    label[i,j] = label[i,j-1];
                elif left == 0 and above != 0:
                    label[i,j] = label[i-1,j];
                else:
                    Lab_prim = Find(label,i,j-1,i-1,j,L);
                    label = Lab_prim;
                    label[i,j] = label[i-1,j];
    return label
def Above_left(i, j, R): #Complementa la funcion Label.
    if i > 0 and j > 0:
        above = R[i-1,j];
        left = R[i,j-1];
    elif i > 0 and j == 0:
        above = R[i-1,j];
        left = 0;
    elif i == 0 and j > 0:
        above = 0;
        left = R[i,j-1];
    else:
        above = 0; 
        left = 0;
    return (above,left)
def Find(L, a, b, c, d, Lw): #Busca elementos en matriz.
    x = L[a,b];
    y = L[c,d];
    find_rc = Find_prim(L,x,Lw);
    row = find_rc[0];
    col = find_rc[1];

    for i in range(0,len(col)):
        aa = row[i];
        bb = col[i];
        L[aa,bb] = y; #cambie item
    return L
def Find_prim(a, b, Lw): #Complemento de la funcion Find.
    #size = a.shape;
    row = np.array([], dtype=np.int64);
    col = np.array([], dtype=np.int64);
    #if len(b) == 1:
    for i in range(0,Lw):
        for j in range(0,Lw):
            if a[i,j] == b:
                row = np.append(row, i);
                col = np.append(col, j);
                #elif len(b) > 1: #Posible trabajo futuro.
                    #size_b = b.shape;
                    #for i in range(0,size[0]):
                        #for j in range(0,size[1]):
                            #for k in range(size_b[1]):
                                #if a[i,j] == b[k]:
                                    #row = np.append(row, i);
                                    #col = np.append(row, j); 
    return [[row],[col]]
def Label(L, R): #Crea la matriz de clusters.
    iD = 1;
    label = np.zeros((L,L));
    for i in range(0,L):
        for j in range(0,L):
            if R[i,j]:
                l_a = Above_left(i,j,R);
                above = l_a[0];
                left = l_a[1];
                d_r =down_right(i, j, R,L-1)
                down = d_r[0];
                right = d_r[1];

                if left == -0.5 and above == -0.5 and down == -0.5 and right == -0.5:
                    label[i,j] = iD;
                    iD = iD + 1;
                #elif left != 1 and above == 0:
                   # label[i,j] = label[i,j-1];
               # elif left == 0 and above != 0:
                    #label[i,j] = label[i-1,j];
                #else:
                   # Lab_prim = Find(label,i,j-1,i-1,j,L);
                    #label = Lab_prim;
                    #label[i,j] = label[i-1,j];
    return label
def Above_left(i, j, R): #Complementa la funcion Label.
    if i > 0 and j > 0:
        above = R[i-1,j];
        left = R[i,j-1];
    elif i > 0 and j == 0:
        above = R[i-1,j];
        left = 0;
    elif i == 0 and j > 0:
        above = 0;
        left = R[i,j-1];
    else:
        above = 0; 
        left = 0;
    return (above,left)
def down_right(i, j, R,L): #Complementa la funcion Label.
    if i < L and j < L:
        d = R[i+1,j];
        r = R[i,j+1];
    elif i <L and j == L:
        d = R[i+1,j];
        r = 0;
    elif i == L and j <L:
        d = 0;
        r = R[i,j+1];
    else:
        d = 0; 
        r = 0;
    return (d,r)


def calcEnergy(config,N):
    '''Energy of a given configuration'''
    E_box = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            S = config[i,j]
            nb = config[(i+1)%N, j] + config[i,(j+1)%N] + config[(i-1)%N, j] + config[i,(j-1)%N]
            E_box[i,j] = (-nb*S)/4.0
    return E_box
    
    
    
def R_matrix(N,R,M):
    x = np.zeros((N,N),dtype= int)
    if M > 0 :
        for i in range(N):
            for j in range(N):
                if R[i,j] == -1 :
                    x[i,j] = 1
    if M < 0 :
        for i in range(N):
            for j in range(N):
                if R[i,j] == 1 :
                    x[i,j] = 1
    if M == 0 :
        print('M=0')
    x = np.vstack ((x, np.zeros(N)) )
    x = np.vstack (( np.zeros(N),x) )
    x = np.column_stack((x, np.zeros(N+2)))
    x = np.column_stack((np.zeros(N+2)  ,x))
    return x
def clustering(N,config,M):
    occu = R_matrix(N,config,M)
    label = np.zeros((N+2,N+2))
    larg_label = 0
    for i in range(1,N+1):
        for j in range(1,N+1):
            min_lab = 0
            max_lab = 0
            if occu [i][j] != 0 :
                left = occu[(i-1)][j]
                above = occu[i][(j-1)]
                if(left == 0 and above == 0):
                    larg_label += 1
                    label[i][j] = larg_label
                elif(left != 0 and above == 0):
                    label[i][j] = label[(i-1)][j] 
                elif(left == 0 and above != 0):
                        label[i][j] = label[i][(j-1)]  
                elif(left != 0 and above != 0):
                    left_lab  = label[(i-1)][j]
                    above_lab = label[i][(j-1)]
                    if(left_lab == above_lab):
                        label[i][j] = left_lab
                    else:
                        min_lab = above_lab
                        max_lab = left_lab
                        if(min_lab > left_lab):
                            min_lab = left_lab
                            max_lab = above_lab
                            for k in range(1,N+1):
                                for l in range(1,N+1):
                                    if label[k][l] == max_lab :
                                        label[k][l] = min_lab
                            label[i][j] = min_lab
    return label
    
    
N       = 256    
mcSteps = 256
T =  np.arange(2.0,2.47,0.01)
T = T[::-1]
print(T)
Si = np.zeros(len(T))
Nu = np.zeros(len(T))
config = initialstate(N)

for tt in range(len(T)):
    iT=1.0/T[tt];
    S1 = 0
    N1 = 0
    #hist = np.zeros(N*N)
    #config = initialstate(N)
    for j in range(mcSteps):
        con = mcmove(config, iT,N,T[tt])
        config = con[0]#initialstate(N)#
        S1 += con[1]
        N1 += con[2]
        print(S1,N1)
    print(tt)
    Si[tt] = S1 / mcSteps
    Nu[tt] = N1 / mcSteps
np.savetxt('T.txt', T)
np.savetxt('S.txt', Si)
np.savetxt('N.txt', Nu)
f = plt.figure(figsize=(18, 10)); # plot the calculated values    
sp =  f.add_subplot(1, 2, 1 );
plt.scatter(T, Si, s=50, marker='o', color='RoyalBlue')
plt.xlabel("Temperature (T)");
plt.ylabel("S ");         plt.axis('tight');
sp =  f.add_subplot(1, 2, 2 );
plt.scatter(T, Nu, s=50, marker='o', color='RoyalBlue')
plt.xlabel("Temperature (T)"); 
plt.ylabel("N ");   plt.axis('tight');
plt.savefig('ssnn.jpg') 
