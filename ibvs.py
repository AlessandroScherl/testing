import numpy as np
import scipy
import math


# image 1000*1000
# principle point (400,400)
# focal length (400, 400)
# pose: r = (0,0,0) => R = I
# T = [10,20,2] == C

# desidred values: (0,0), (800,0), (0,800), (800,800)
# measured values: (50,50), (850, 50), (50, 850), (850,850)
# assume Z = 50

s_uv = np.array([[50, 50],
                 [850,50],
                 [50,850],
                 [850,850]]
                 )

s_star_uv = np.array([[0, 0],
                      [800,0],
                      [0,800],
                      [800,800]]
                      )

s_xy = np.zeros([4, 2], dtype=int)
s_star_xy = np.zeros([4, 2], dtype=int)

## 1.) calculate x ,y 

for count in range (len(s_star_uv)):
    
    s_xy[count,0] = ((s_uv[count,0]-400)/400)
    s_xy[count,1] = ((s_uv[count,1]-400)/400)

    s_star_xy[count,0] = ((s_star_uv[count,0]-400)/400)
    s_star_xy[count,1] = ((s_star_uv[count,1]-400)/400)

#print(s_xy)
#print(s_star_xy)

e = s_xy - s_star_xy

#print(e)
Z = 50
x = s_xy[0,0]
y = s_xy[0,1]

L1 = np.array([[-1/Z, 0,    s_xy[0,0]/Z, s_xy[0,0]*y,   -(1+s_xy[0,0]*s_xy[0,0]), s_xy[0,1]],
              [ 0,   -1/Z, s_xy[0,1]/Z, 1+s_xy[0,1]*s_xy[0,1], -s_xy[0,0]*s_xy[0,1],     -s_xy[0,0]]])

L2 = np.array([[-1/Z, 0,    s_xy[1,0]/Z, s_xy[1,0]*y,   -(1+s_xy[1,0]*s_xy[1,0]), s_xy[1,1]],
              [ 0,   -1/Z, s_xy[1,1]/Z, 1+s_xy[1,1]*s_xy[1,1], -s_xy[1,0]*s_xy[1,1],     -s_xy[1,0]]])

L3 = np.array([[-1/Z, 0,    s_xy[2,0]/Z, s_xy[2,0]*y,   -(1+s_xy[2,0]*s_xy[2,0]), s_xy[2,1]],
              [ 0,   -1/Z, s_xy[2,1]/Z, 1+s_xy[2,1]*s_xy[2,1], -s_xy[2,0]*s_xy[2,1],     -s_xy[2,0]]])

L4 = np.array([[-1/Z, 0,    s_xy[3,0]/Z, s_xy[3,0]*y,   -(1+s_xy[3,0]*s_xy[3,0]), s_xy[3,1]],
              [ 0,   -1/Z, s_xy[3,1]/Z, 1+s_xy[3,1]*s_xy[3,1], -s_xy[3,0]*s_xy[3,1],     -s_xy[3,0]]])

L = np.concatenate([L1,L2,L3,L4])

#print(L)

L_p = np.linalg.pinv(L)
e_new = np.array([[1,1,0,1,1,0,0,0]])

print(L_p.shape)
#print(e.shape)
#print(e)
print(np.transpose(e_new).shape)

v_c = L_p @ np.transpose(e_new)  # @ == matmul

print(v_c)
