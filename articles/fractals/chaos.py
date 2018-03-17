class Chaos:
    """
    Models the dynamical system with :math:`x_{t+1} = r x_t (1 - x_t)`
    """
    def __init__(self, x0, r):
        """
        Initialize with state x0 and parameter r 
        """
        self.x, self.r = x0, r
        
    def update(self):
        "Apply the map to update state."
        self.x =  self.r * self.x *(1 - self.x)
        
    def generate_sequence(self, n):
        "Generate and return a sequence of length n."
        path = []
        for i in range(n):
            path.append(self.x)
            self.update()
        return path


import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ch = Chaos(0.1, 4)
r = 2.5
while r < 4:
    ch.r = r
    t = ch.generate_sequence(5000)[4990:]
    ax.plot([r] * len(t), t, 'k.', ms=0.06)
    r = r + 0.001

ax.set_xlabel(r'$r$', fontsize=16)
ax.set_ylabel(r'$x$', fontsize=16)
plt.savefig("bifurcation.png",bbox_inches='tight',dpi=300)
plt.show()        

#plt.xlim([3.3,3.75])
#plt.ylim([0.7,0.98])
#plt.savefig("bifurcation_zoom.png",bbox_inches='tight',dpi=300)
#plt.show()