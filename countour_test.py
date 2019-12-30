import numpy as np
import matplotlib.pyplot as plt

def _cvxopt(x, z, y, s, G, g, C, d, A, b, eta, maxk):
    
    path = []
    path.append(x)
    
    #compute the residuals
    mA, nA = A.shape[0], A.shape[1]
    
    mC, nC = C.shape[0], C.shape[1]
    
    e = np.matrix(np.ones((nC,1)))
    
    rL = G*x + g - A*y - C*z
    rA = -A.T*x + b
    rC = -C.T*x + s + d
    rsz = np.multiply(s,z)
    mu = np.sum(np.multiply(z,s))/nC
    
    # number of iteration, epsilon, tolerance
    
    k = 0
    #maxk = 2000
    eps_L = 1e-10
    eps_A = 1e-10
    eps_C = 1e-10
    eps_mu = 1e-10
    
    while (k <= maxk):# and np.linalg.norm(rL)>=eps_L and np.linalg.norm(rA)>=eps_A and np.linalg.norm(rC)>=eps_C and abs(rsz).all()>=eps_mu):
    
        
        # solve the sytem of equation : predictor step
        
        lhs = np.block([[G,-A,-C],[-A.T,np.zeros((nA,nA)),np.zeros((nA,nC))],[-C.T,np.zeros((nC,nA)),-np.diag((s/z).A1)]])
        
        #L, D, P = ldl(lhs)
        
        #L, D, P = np.matrix(L), np.matrix(D), np.matrix(np.eye(len(L))[:,P])
        
        rhs = np.block([[-rL],[-rA],[-rC + rsz/z]])
        
        dxyz_a = np.linalg.solve(lhs,rhs)
        #dxyz_a = P*(np.linalg.inv(L.T)*(np.linalg.inv(D)*(np.linalg.inv(L)*(P.T*rhs))))
        
        dx_a = dxyz_a[0:len(x)]
        dy_a = dxyz_a[len(x):len(y)+len(x)]
        dz_a = dxyz_a[len(y)+len(x):len(x)+len(y)+len(z)]        
        ds_a = -((rsz + np.multiply(s,dz_a))/z)
        
        # compute alpha_aff
        
        alpha_a = 1
        
        idx_z = np.nonzero(dz_a < 0)
        
        if (len(idx_z[0])!=0):
            alpha_a =  min(alpha_a, np.min(-z[idx_z]/dz_a[idx_z]))
            
        idx_s = np.where(ds_a < 0)
        
        if (len(idx_s[0])!=0):
            alpha_a =  min(alpha_a, np.min(-s[idx_s]/ds_a[idx_s]))
        
        mu_a = ((z + alpha_a*dz_a).T*(s + alpha_a*ds_a))/nC
        
        sigma = np.linalg.matrix_power(mu_a/mu,3)
        
    
        # solve the sytem of equation : corrector step
        
        rsz = rsz + np.multiply(ds_a,dz_a) - sigma[0,0]*mu*e
        rhs = np.block([[-rL],[-rA],[-rC + rsz/z]])
        dxyz = np.linalg.solve(lhs,rhs)
        #dxyz = P*(np.linalg.inv(L.T)*(np.linalg.inv(D)*(np.linalg.inv(L)*(P.T*rhs))))
        
        dx = dxyz[0:len(x)]
        dy = dxyz[len(x):len(y) + len(x)]
        dz = dxyz[len(y) + len(x):len(x) + len(y) + len(z)]
        ds = -((rsz + np.multiply(s,dz))/z)        
        
        # compute alpha
        
        alpha = 1
        
        idx_z = np.where(dz < 0)
        
        if (len(idx_z[0])!=0):
            alpha =  min(alpha, np.min(-z[idx_z]/dz[idx_z]))
            
        idx_s = np.where(ds < 0)
        
        if (len(idx_s[0])!=0):
            alpha =  min(alpha, np.min(-s[idx_s]/ds[idx_s]))
            
        # Update x, z, s
        
        #print(alpha)
        x = x + eta*np.multiply(alpha,dx)
        z = z + eta*np.multiply(alpha,dz)
        y = y + eta*np.multiply(alpha,dy)
        s = s + eta*np.multiply(alpha,ds)
        
        k = k+1
    
        # Update rhs and mu
        
        rL = G*x + g - A*y - C*z
        rA = -A.T*x + b
        rC = -C.T*x + s + d
        rsz = np.multiply(s,z)
        mu = np.sum(np.multiply(z,s))/nC
        
        path.append(x)
        
    return x, path, k



G = np.matrix([[8, 2],[2,2]])
g = np.matrix([[2],[3]])
C = np.matrix([[1,-1,-1],[-1,-1,0]])
x, y = np.matrix([[6],[1]]), np.matrix([[1]]) 
z, s = np.matrix([[1],[1],[1]]),np.matrix([[1],[1],[1]])
d = np.matrix([[0],[-4],[-3]])
A = np.matrix([[1],[0]])
b = np.matrix([[-2]])

eta = 0.5
maxk = 10
w, path, k = _cvxopt(x, z, y, s, G, g, C, d, A, b, eta, maxk)

pathList = []

for ele in path: 
    pathList.append([ele[0,0],ele[1,0]])
    
pathListStacked = np.vstack(pathList)

x = np.linspace(0, 3, 100)
y = np.linspace(0, 3, 100)

x2 = np.linspace(3,3, 1000)

y2 = np.linspace(-4,8, 1000)
x3 = np.linspace(-2,-2, 1000)
y3 = np.linspace(-4,8, 1000)


def f(x1,x2,mu):
    
    #f = 2*x1 + 3*x2 + mu*(np.log(21 - 7*x1 - 3*x2) + np.log(21 - 3*x1 - 7*x2))
    f = 2*x1 + 3*x2 + mu*(np.log(x1) + np.log(x2) + np.log(21 - 7*x1 - 3*x2) + np.log(21 - 3*x1 - 7*x2))
    
    return f


c1 = x
c2 = -x + 4
c3 = (1 - x)
X, Y = np.meshgrid(x, y)
Z = f(X, Y, 1000)
plt.figure(figsize=(16,12))
plt.contour(X, Y, Z, 500)


eta = 0.5
maxk = 10
w, path, k = _cvxopt(x, z, y, s, G, g, C, d, A, b, eta, maxk)

pathList = []

for ele in path: 
    pathList.append([ele[0,0],ele[1,0]])
    
pathListStacked = np.vstack(pathList)
contours = []

for mu in range(10,0,0.1):
    
    




plt.figure(figsize=(8,6))
plt.plot(pathListStacked[:,0],pathListStacked[:,1],'--')
plt.plot(x,c1)
plt.plot(x,c2)
plt.plot(x2,y2)
plt.plot(x3,y3, color = 'b')
plt.scatter(pathListStacked[len(pathListStacked)-1,0],pathListStacked[len(pathListStacked)-1,1])
plt.contour(X, Y, Z, 20)