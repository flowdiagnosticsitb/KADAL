import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import scipy
import warnings
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time
warnings.filterwarnings("ignore")


class Conduction:
    """
    Solver for 2D heat conduction with heat source on the plate and
    Gaussian random field as the conductivity coefficient.
    """

    def __init__(self, x=(-0.5, 0.5), y=(-0.5, 0.5)):
        """
        Initialize the variables

        Args:
            x (tuple): tuple of x coordinates (x0,x1)
            y (tuple): tuple of x coordinates (y0,y1)
        """
        self.ndimen = 2
        self.x0 = x[0]
        self.y0 = y[0]
        self.x1 = x[1]
        self.y1 = y[1]
        self.length = self.x1 - self.x0
        self.width = self.y1 - self.y0

    def run(self,xi,view=False):
        """
        Wrapper to run the code

        Arg:
            - xi (nparray): array of input

        Return:
            - tavgB (float): Average temperature inside region B
        """
        gridx, gridy = self.creategrid(100, 100, view=False)
        gx = self.calculatealpha(xi, gridx, gridy, 0.2, view=False)
        self.creatematrix(gx)
        self.solve(view=view)
        tavgB = self.calcB()
        return tavgB

    def creatematrix(self, kx):
        matsize = np.size(self.z,0)
        self.coeffmat = np.zeros(shape=[matsize,matsize])

        # Create lower neumann BC
        for i in range(self.nx+1):
            self.coeffmat[i,i] = 1
            self.coeffmat[i,i+self.nx+1] = -1

        # Create upper dirichlet BC
        for i in range(-(self.nx+1),0):
            self.coeffmat[i,i] = 1

        # Create the remaining
        for i in range(self.nx+1,matsize-(self.nx+1)):
            if i % (self.nx+1) == 0: # left neumann BC
                self.coeffmat[i, i] = 1
                self.coeffmat[i, i + 1] = -1
            elif i % (self.nx+1) == self.nx: # right neumann BC
                self.coeffmat[i, i] = 1
                self.coeffmat[i, i - 1] = -1
            else:
                iloc,jloc = self.z[i,:]
                iloc = int((np.round(iloc,2) + 0.5)*100)
                jloc = int((np.round(jloc,2) + 0.5)*100)
                k1 = (kx[jloc,iloc+1] - kx[jloc,iloc+1]) / ((2*self.dx)**2)
                k2 = (kx[jloc+1,iloc] - kx[jloc-1,iloc]) / ((2*self.dy)**2)
                k3 = kx[jloc,iloc] / (self.dx**2)
                k4 = kx[jloc, iloc] / (self.dy ** 2)
                self.coeffmat[i,i] = 2*(k3+k4)
                self.coeffmat[i, i - 1] = -(k3-k1)
                self.coeffmat[i, i + 1] = -(k3+k1)
                self.coeffmat[i, i - (self.nx + 1)] = -(k4-k2)
                self.coeffmat[i, i + (self.nx + 1)] = -(k4+k2)

    def solve(self, view=False):
        self.source = np.zeros(shape=[np.size(self.z,0)])
        for i in range(np.size(self.z,0)):
            x = self.z[i,0]
            y = self.z[i,1]
            if (x >= 0.2 and x <= 0.3) and ((y >= 0.2 and y <= 0.3)):
                self.source[i] = 2000
            else:
                pass

        self.tdist = np.linalg.solve(self.coeffmat,self.source)
        tdist = self.tdist.reshape((self.nx+1, self.ny+1))
        if view:
            fig, ax = plt.subplots()
            surf = ax.imshow(tdist, cmap=cm.jet, extent=[-0.5, 0.5, -0.5, 0.5], origin='lower', interpolation='bilinear')
            clb = fig.colorbar(surf)
            clb.ax.set_title('T[\u2103]')
            plt.show()

    def calcB(self):
        blist = []
        for i in range(np.size(self.z,0)):
            x = self.z[i, 0]
            y = self.z[i, 1]
            if (x >= -0.3 and x <= -0.2) and ((y >= -0.3 and y <= -0.2)):
                blist.append(self.tdist[i])
            else:
                pass
        blist = np.array(blist)
        TavgB = np.sum(blist*(0.01**2)) / 0.01
        return TavgB


    def calculatealpha(self, xi, gridx, gridy, theta=0.2, view=False):
        grfx, grfy = self.rndfgrid()
        M, li, phii = self.grandomfield(theta,grfx,grfy)
        self.z = np.hstack((gridx.reshape(self.nn, 1), gridy.reshape(self.nn, 1)))
        gx = np.zeros(shape=[np.size(self.z,0)])
        for i in range(np.size(self.z,0)):
            gx[i] = self.calcgz(self.z[i,:],M,xi,li,phii)
        gx = gx.reshape((np.size(gridx,0), np.size(gridx,1)))

        if view:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            surf = ax.plot_surface(gridx, gridy,gx,cmap=cm.coolwarm, linewidth=0, antialiased=False)
            plt.show()

        kx = np.exp(1 + gx*0.3)
        return kx

    def creategrid(self, nx=100, ny=100, view=False):
        """
        Create discretization for finite difference heat equation solver.

        Args:
             - nx (int): number of spacing on X direction. Default to 100. (number of points is nx+1)
             - ny (int): number of spacing on Y direction. Default to 100. (number of points is ny+1)
             - view (bool): visualize grid or not. Default to False.

         Returns:
             - gridx (nparray): x coordinates for each point across the space.
             - gridy (nparray): y coordinates for each point across the space.
        """
        self.nx = nx
        self.ny = ny
        xx = np.linspace(self.x0, self.x1, nx + 1)
        yy = np.linspace(self.y0, self.y1, ny + 1)
        self.nn = (nx + 1) * (ny + 1)
        gridx, gridy = np.meshgrid(xx, yy)
        self.dx = self.length / nx
        self.dy = self.width / ny

        if view is True:
            for i in range(nx + 1):
                temp1 = np.array([[-0.5, yy[i]], [0.5, yy[i]]])
                temp2 = np.array([[xx[i], -0.5], [xx[i], 0.5]])
                plt.plot(temp1[:, 0], temp1[:, 1], 'k')
                plt.plot(temp2[:, 0], temp2[:, 1], 'k')
            # plt.scatter(gridx, gridy)
            plt.show()

        return gridx, gridy

    def rndfgrid(self, nx=10, ny=10, view=False):
        """
        Create RGF nodes in the domain.

        Args:
            nx (int): number of spacing on X direction. Default to 10. (number of points is nx+1)
            ny (int): number of spacing on Y direction. Default to 10. (number of points is ny+1)
            view (bool): display nodes or not.

        Returns:
             - rf_gridx (nparray): x coordinates for each point across the space.
             - rf_gridy (nparray): y coordinates for each point across the space.
        """
        rf_xx = np.linspace(self.x0, self.x1, nx+1)
        rf_yy = np.linspace(self.y0, self.y1, ny+1)
        self.rf_nn = (nx+1) * (ny+1)
        rf_gridx, rf_gridy = np.meshgrid(rf_xx, rf_yy)

        if view is True:
            for i in range(nx+1):
                temp1 = np.array([[-0.5, rf_yy[i]], [self.rf_yy[i]]])
                temp2 = np.array([[rf_xx[i], -0.5], [rf_xx[i], 0.5]])
                plt.plot(temp1[:, 0], temp1[:, 1], 'k')
                plt.plot(temp2[:, 0], temp2[:, 1], 'k')
            plt.scatter(rf_gridx, rf_gridy)
            plt.show()

        return rf_gridx,rf_gridy

    def grandomfield(self, theta, rf_gridx, rf_gridy):
        """
        Calculate the components inside the Gaussian random field.

        Args:
            - theta (float): Lengthscale of Kernel function
            - rf_gridx (nparray):  x coordinates for each point of GRF grid across the space.
            - rf_gridy (nparray):  y coordinates for each point of GRF grid across the space.

        Returns:
            - M (int): Number of dimension of the input variables.
            - li (nparray): Eigenvalues of correlation matrix.
            - phii (nparray): Eigenvectors of correlation matrix.
        """
        self.theta = theta * np.ones(shape=[self.ndimen])
        self.zeta = np.hstack((rf_gridx.reshape(self.rf_nn, 1), rf_gridy.reshape(self.rf_nn, 1,)))
        c_zetazeta = kernel(self.zeta, self.zeta, self.ndimen, self.theta)
        li, phii = np.linalg.eigh(c_zetazeta)
        li = np.flip(li,0)
        phii = np.flip(phii,1)

        for M in range(1,self.rf_nn+1):
            temp1 = np.sum(li[:M]) / np.sum(li)
            if temp1 >= 0.99:
                break
            else:
                pass

        return M,li,phii

    def calcgz(self,z,M,xi,li,phii):
        c_zzeta = kernel(z, self.zeta, self.ndimen, self.theta).transpose()
        gztemp = np.zeros(shape=[M])
        for i in range(M):
            gztemp[i] = (xi[i]/np.sqrt(li[i])) * np.dot(phii[:,i], c_zzeta)
        gz = np.sum(gztemp)
        return gz

    def basisfunc(self,z,i,phii):
        c_zzeta = kernel(z, self.zeta, self.ndimen, self.theta).transpose()
        gztemp = np.dot(phii[:,i], c_zzeta)
        return gztemp

def kernel(XN, XM, nvar, theta):
    if XN.ndim == 1:
        XN = np.array([XN])
    mdist = np.zeros((np.size(XN, 0), np.size(XM, 0), nvar))
    for ii in range(0, nvar):
        X1 = np.transpose(np.array([XN[:, ii]]))
        X2 = np.transpose(np.array([XM[:, ii]]))
        mdist[:, :, ii] = (cdist(X1, X2, 'euclidean') ** 2) / (theta[ii] ** 2)
    Psi = np.exp(-1 * np.sum(mdist, 2))
    return Psi

if __name__ == "__main__":
    case = 1
    if case == 1:
        test = Conduction()
        gridx, gridy = test.creategrid(100, 100, view=False)
        grfx, grfy = test.rndfgrid()
        z = np.hstack((gridx.reshape(test.nn, 1), gridy.reshape(test.nn, 1)))
        M, li, phii = test.grandomfield(0.2, grfx, grfy)
        gx = np.zeros(shape=[np.size(z, 0)])
        fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(11, 11), subplot_kw={'xticks': [], 'yticks': []})
        for j in range(20):
            ax = axs.flat[j]
            for i in range(np.size(z, 0)):
                gx[i] = test.basisfunc(z[i, :], j, phii)
            gxi = gx.reshape((np.size(gridx,0), np.size(gridx,1)))
            surf = ax.imshow(gxi, cmap=cm.jet, extent=[-0.5, 0.5, -0.5, 0.5], origin='lower', interpolation='bilinear')
        plt.tight_layout()
        plt.show()
    else:
        xi = 1*np.random.randn(53)
        t = time.time()
        plate = Conduction()
        tavgB = plate.run(xi, view=True)
        print(tavgB)
        elapsed = time.time() - t
        print(elapsed,'s')


