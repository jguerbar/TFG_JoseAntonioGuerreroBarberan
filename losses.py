import torch
import torch.nn as nn
from torch.nn.functional import mse_loss

class PolymorphicMSE_loss(nn.Module):
	def __init__(self):
		super(PolymorphicMSE_loss, self).__init__()
	def forward(self, x, y, _):
		return mse_loss(x,y)

class L1_Charbonnier_loss(nn.Module):
	def __init__(self):
		super(L1_Charbonnier_loss, self).__init__()
		self.eps = 1e-6
	def forward(self, X, Y):
		diff = torch.add(X, -Y)
		error = torch.sqrt( diff * diff + self.eps )
		loss = torch.mean(error) 
		return loss


class Navier_Stokes_informed_loss(nn.Module):
	def __init__(self, rho=1, mu=0.00001, dt=1/32,dx=1/255,dy=1/255):
		super(Navier_Stokes_informed_loss, self).__init__()
		self.rho = rho
		self.mu = mu
		self.dt = dt
		self.dx = dx
		self.dy = dy
	def forward(self, x, _, std):
		#De-normalize to recover true physical values
		x *= std

		u = x[:,0,:,:]
		v = x[:,1,:,:]
		p = x[:,2,:,:]

		#Calculate forward difference for time and central difference for space
		dudt = (u-torch.roll(u,1,dims=0))/self.dt
		dvdt = (v-torch.roll(v,1,dims=0))/self.dt
		#dpdt = (p-torch.roll(p,1,dims=0))[1:,:,:]

		#Central first order difference
		dudx = (torch.roll(u,-1,dims=1)-torch.roll(u,1,dims=1))/(2*self.dx)
		dudy = (torch.roll(u,-1,dims=2)-torch.roll(u,1,dims=2))/(2*self.dy)
		dvdx = (torch.roll(v,-1,dims=1)-torch.roll(v,1,dims=1))/(2*self.dx)
		dvdy = (torch.roll(v,-1,dims=2)-torch.roll(v,1,dims=2))/(2*self.dy)
		dpdx = (torch.roll(p,-1,dims=1)-torch.roll(p,1,dims=1))/(2*self.dx)
		dpdy = (torch.roll(p,-1,dims=1)-torch.roll(p,1,dims=2))/(2*self.dy)

		#Central second order difference
		d2udx2 = (torch.roll(u,-1,dims=1)-2*u+torch.roll(u,1,dims=1))/self.dx
		d2udy2 = (torch.roll(u,-1,dims=2)-2*v+torch.roll(u,1,dims=2))/self.dy
		d2vdx2 = (torch.roll(v,1,dims=1)-2*v+torch.roll(v,1,dims=1))/self.dx
		d2vdy2 = (torch.roll(v,1,dims=2)-2*v+torch.roll(v,1,dims=2))/self.dy

		#Bordes are not computed, as it's finite differences makes no sense considering torch.roll method
		u = u[1:,1:-1,1:-1]
		v = v[1:,1:-1,1:-1]
		p = p[1:,1:-1,1:-1]
		dudt = dudt[1:,1:-1,1:-1]
		dudx = dudx[1:,1:-1,1:-1]
		dudy = dudy[1:,1:-1,1:-1]
		dvdt = dvdt[1:,1:-1,1:-1]
		dvdx = dvdx[1:,1:-1,1:-1]
		dvdy = dvdy[1:,1:-1,1:-1]
		#dpdt = dpdt[1:,1:1,1:-1]
		dpdx = dpdx[1:,1:-1,1:-1]
		dpdy = dpdy[1:,1:-1,1:-1]

		d2udx2 = d2udx2[1:,1:-1,1:-1]
		d2udy2 = d2udy2[1:,1:-1,1:-1]
		d2vdx2 = d2vdx2[1:,1:-1,1:-1]
		d2vdy2 = d2vdy2[1:,1:-1,1:-1]
		#dp2dx2 = dp2dx2[1:,2:,2:]
		#dp2dy2 = dp2dy2[1:,2:,2:]

		#Calculate residuals
		#Continuity eq
		r0 = torch.abs(dudx+dvdy)
		#Momentum
		r1 = torch.abs((1/self.rho)*dudt+u*dudx+v*dudy + dpdx - self.mu*(d2udx2+d2udy2))
		r2 = torch.abs((1/self.rho)*dvdt+u*dvdx+v*dvdy + dpdy - self.mu*(d2vdx2+d2vdy2))
		
		return 0.01*(torch.mean(r0)+torch.mean(r1)+torch.mean(r2))/3
