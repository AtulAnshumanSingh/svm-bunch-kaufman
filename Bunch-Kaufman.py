import numpy as np
import copy

def bunch_kaufman(A):
    
    N = A.shape[0]
    
    for index in range(0,N):
        
        A[index,index+1:] = 0
    
    L = np.matrix(np.eye(N,N))
    
    alpha = (1 + np.sqrt(17))/8
    
    pivot = np.zeros(N)
    
    P = np.arange(1,N + 1,1)

    k = 0
    
    while k <= N-2:
        
        lambda_1, j = np.max(np.abs(A[k:N,k])), np.argmax(np.abs(A[k:N,k]))
        
        r = k + j
        
        if np.abs(A[k,k]) >= alpha * lambda_1:
            
            L[k+1:N, k] = A[k+1:N, k].copy()/A[k,k]
            A[k+1:N,k+1:N] = A[k+1:N, k+1:N] - L[k+1:N, k]*A[k+1:N, k].T
            A[k+1:N,k] = np.matrix(np.zeros(N - (k + 1))).T
            
            pivot[k] = 1
            k = k + 1
            
        else:
            
            if r < N - 1:
            
                lambda_r = np.max([np.max(np.abs(A[r,k:r])), np.max(np.abs(A[r+1:N,r]))])
        
            else:
                
                lambda_r = np.max(np.abs(A[r,k:r]))
                
       
            if np.abs(A[k,k]) * lambda_r >= alpha * lambda_1 * lambda_1:
                            
                L[k+1:N, k] = A[k+1:N, k].copy()/A[k,k]
                A[k+1:N,k+1:N] = A[k+1:N, k+1:N] - L[k+1:N, k]*A[k+1:N, k].T
                A[k+1:N,k] = np.matrix(np.zeros(N - (k + 1))).T
            
                pivot[k] = 1
                k = k + 1

                
            else:
                
                if np.abs(A[r,r]) >= alpha*lambda_r:
                    
                    P[k], P[r] = P[r], P[k].copy()
                    
                    A[k,k], A[r,r] = A[r,r], A[k,k].copy()
                
                    A[r+1:N, r], A[r+1:N, k] = A[r+1:N, k], A[r+1:N, r].copy()
                    
                    A[k+1:r, k], A[r, k+1:r] = np.transpose(A[r, k+1:r]), np.transpose(A[k+1:r, k].copy())

                    if k > 0:

                        L[k, 0:k], L[r, 0:k] = L[r, 0:k], L[k, 0:k].copy()
                        
                    L[k+1:N, k] = A[k+1:N, k].copy()/A[k,k]
                    A[k+1:N,k+1:N] = A[k+1:N, k+1:N] - L[k+1:N, k]*A[k+1:N, k].T
                    A[k+1:N,k] = np.matrix(np.zeros(N - (k + 1))).T

                    pivot[k] = 1
                    k = k + 1
                    
                else:
                    
                    if np.abs(A[r,r]) < alpha * lambda_r:
                        
                        P[k + 1], P[r] = P[r], P[k + 1].copy()
                        
                        A[k+1,k+1], A[r,r] = A[r,r], A[k+1,k+1].copy()
                        
                        A[r+1:N,k+1], A[r+1:N,r] = A[r+1:N,r], A[r+1:N,k+1].copy()
                        
                        A[k+1,k], A[r,k] = A[r,k], A[k+1,k].copy()
                        
                        A[k+2:r,k+1], A[r,k+2:r] = np.transpose(A[r,k+2:r]), np.transpose(A[k+2:r,k+1].copy())
                        
                        if k > 0:
                            
                            L[k+1,1:k], L[r, 1:k] = L[r, 1:k], L[k+1,1:k].copy()

                        E = np.eye(2,2)
                        
                        E[0,0] = A[k,k]
                        E[1,0] = A[k+1,k]
                        E[0,1] = E[1,0]
                        E[1,1] = A[k+1,k+1]
                    
                        detE = E[0,0]*E[1,1] - E[0,1]*E[1,0]
                        
                        invE = np.array([[E[1,1], -E[1,0]], [-E[1,0],E[0 ,0]]])/detE
                        
                        L[k+2:N, k:k+2] = np.matmul(A[k+2:N,k:k+2].copy(),invE)
                        A[k+2:N, k+2:N] = A[k+2:N, k+2:N] - L[k+2:N, k:k+2]*A[k+2:N, k:k+2].T
                        A[k+2:N, k] = np.matrix(np.zeros(N - (k + 2))).T
                        A[k+2:N, k+1] = np.matrix(np.zeros(N - (k + 2))).T
                        
                        pivot[k] = 2
                        
                        k = k + 2
                        
        
    if 0 == pivot[N - 1]:
        if 1 == pivot[N - 2]:
            pivot[N - 1] = 1
            
    
    for ind in range(0,N - 1):

        A[ind, ind + 1:] = 0
                
    return A, L, P - 1, pivot


def solve(A, P, b):
    
    N = A.shape[0]
    
    i = 0
    
    while i < N - 1:
        
        if i < N:
                 
            ip1 = i + 1
            save = b[P[i]]
            
            if P[ip1] > 0:
                
                print("This Case")
                
                b[P[i]] = b[i]
                if A[i,i] == 0:
                    print("Fail")
                
                b[i] = save / A[i,i]
                
                for j in range(ip1, N):
                    
                    b[j] = b[j] + A[i,j]*save
                
                i = ip1
                
            else:
            
                temp = b[i]
                b[P[i]] = b[ip1]
                det = P[ip1]
                
                b[i] = (temp * A[ip1,ip1] - save * A[i,ip1])/det
                b[ip1] = (save * A[i,i] - temp * A[i,ip1])/det
                
                for j in range(i + 2, N):
                    
                    b[j] = b[j] + A[i,j] * temp + A[ip1, j] * save
                
                i = i + 2
        
    if i == N - 1:
        
        if A[i,i] == 0:
            
            raise "Fail"
        
        b[i] = b[i]/A[i,i]
        
        i = N - 1
        
    else: 
        
        i = N - 2

    
    while i > 0:
        
        if i > 0:
            
            if P[i] < 0:
                
                ii = i - 1
            
            else:
                
                ii = i
                
        for k in range(ii,i):
            
            save = b[k]
            
            for j in range(i + 1, N):
                
                save = save + A[k,j] * b[j]
            
            b[k] = save
        
        b[i] = b[P[ii]]
        b[P[ii]] = save
        
        i = ii - 1
    
    return b
    
    
solve(A, P, b)
      
#Example 1
A = np.matrix([[6,12,3,-6],
              [12,-8,-13,4],
              [3,-13,-7,1],
              [-6,4,1,6]],dtype = np.float32)
    
    
A_ = np.matrix([[6,12,3,-6],
              [12,-8,-13,4],
              [3,-13,-7,1],
              [-6,4,1,6]],dtype = np.float32)
    
#Example 2    
A = np.matrix([[-3,-3,-18,-30,18],
              [-3,-1,-4,-48,8],
              [-18,-4,-6,-274,6],
              [-30,-48,-274,119,19],
              [18, 8, 6, 19, 216]],dtype = np.float32)
    
    
A_ = np.matrix([[-3,-3,-18,-30,18],
              [-3,-1,-4,-48,8],
              [-18,-4,-6,-274,6],
              [-30,-48,-274,119,19],
              [18, 8, 6, 19, 216]],dtype = np.float32)

#Example 3
A = np.matrix([[-4,0,-16,-32,28],
              [0,1,5,10,-6],
              [-16,5,-37,-66,64],
              [-32,10,-66,-85,53],
              [28,-6,64,53,-15]],dtype = np.float32)
    
A_ = np.matrix([[-4,0,-16,-32,28],
              [0,1,5,10,-6],
              [-16,5,-37,-66,64],
              [-32,10,-66,-85,53],
              [28,-6,64,53,-15]],dtype = np.float32)
    
    
    
    
D, L, P, pivot = bunch_kaufman(A)

L = L + L.T

'''
for index in range(0, 5):
    
    L[index,index] = D[index,index] 
'''

#Example 2
b = np.matrix([327, 291, 1290, 275, 1720]).T

#Example 3
b = np.matrix([448, -111, 1029, 1207, -719]).T

P_ = np.matrix(np.eye(5,5)[:,P])

LHS = b[P]

y = np.linalg.solve(L*D,LHS)

L_ = L*D

LHS = LHS/L[0,0]

for i in range(1,len(L_)):
    
    LHS = LHS - 

x = np.linalg.solve(L.T,y)

np.round(L*D*L.T)

dxyz = P_*(np.linalg.inv(L*D*L.T)*(P_.T*b))

_A = copy.deepcopy(L)

solve(_A, P, copy.deepcopy(b))