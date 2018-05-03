import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter,MultipleLocator, FormatStrFormatter
from pylab import *
import time

def julia(c,xmin=-2,xmax=1,ymin=-1.1667,ymax=1.1667,max_iters=1000,width=2250,height=1500):

 mat=np.zeros((height,width))

 x=np.linspace(xmin,xmax,width)
 y=np.linspace(ymin,ymax,height)

 for j in range(height):
  for i in range(width):
    c=complex(x[i],y[j])  #mandelbrot
    #z=complex(x[i],y[j]) #julia
    z=complex(0,0)  #mandelbrot inital z
    #smoothcolor = e**(-abs(z)) #smooth color julia
    for iters in range(max_iters):
      z=z*z+c
      #smoothcolor = smoothcolor + e**(-abs(z))   #smooth color julia
      if ((z.real**2+z.imag**2)>=30): 
         nsmooth = iters + 1 - log(log(abs(z)))/log(2)  #mandelbrot smooth color
#      	 mat[j,i]=(iters+1)  # no smoothing
#     	 mat[j,i]=smoothcolor/max_iters  # smooth color julia
# 	 mat[j,i]=1  # black/white
	 mat[j,i]=nsmooth  #smooth color mandelbrot
	 break

 figure(num=None, figsize=(15, 10),facecolor='white')
 ax=subplot(111)
 cmap=matplotlib.cm.jet
 cmap.set_bad('k')
 im=plt.imshow(log(mat),cmap=cmap,extent=(xmin,xmax,ymin,ymax))
 ax.yaxis.set_major_formatter( NullFormatter() )
 ax.xaxis.set_major_formatter( NullFormatter() )
 plt.savefig('mandelbrot.png',dpi=72,bbox_inches='tight')
 plt.show()

t1=time.clock()
#julia(complex(-0.4,-0.6),complex(-0.4,-0.6),0.2,0.2)
#julia(complex(-0.74,0.11))
#julia(complex(-0.62772,0.42193))
#julia(complex(0.285,0.013))
julia(complex(0,0))
t2=time.clock()
print t2-t1