import torch
import torch.nn as nn
from torch.nn.functional import mse_loss

class MSE_loss(nn.Module):
	def __init__(self):
		super(MSE_loss, self).__init__()
	def forward(self, x, y, _):
		return mse_loss(x,y)
class MaxLoss(nn.Module):
	def __init__(self):
		super(MaxLoss, self).__init__()
	def forward(self, x, y, std):
		x *= std
		y *= std
		return torch.max(torch.abs(x-y))
class Navier_Stokes_informed_loss(nn.Module):
	def __init__(self, rho=1, mu=0.00001, dt=1/32,dx=1/255,dy=1/255):
		super(Navier_Stokes_informed_loss, self).__init__()
		self.rho = rho
		self.mu = mu
		self.dt = dt
		self.dx = dx
		self.dy = dy
	def calculate_res(self,x,std):
		#De-normalize to recover true physical values
		x *= std
		u = x[:,0,:,:]
		v = x[:,1,:,:]
		p = x[:,2,:,:]

		#Central first order difference
		dudt = (torch.roll(u,-1,dims=0)-torch.roll(u,1,dims=0))/self.dt
		dvdt = (torch.roll(v,-1,dims=0)-torch.roll(v,1,dims=0))/self.dt
		dudx = (torch.roll(u,-1,dims=1)-torch.roll(u,1,dims=1))/(2*self.dx)
		dudy = (torch.roll(u,-1,dims=2)-torch.roll(u,1,dims=2))/(2*self.dy)
		dvdx = (torch.roll(v,-1,dims=1)-torch.roll(v,1,dims=1))/(2*self.dx)
		dvdy = (torch.roll(v,-1,dims=2)-torch.roll(v,1,dims=2))/(2*self.dy)
		dpdx = (torch.roll(p,-1,dims=1)-torch.roll(p,1,dims=1))/(2*self.dx)
		dpdy = (torch.roll(p,-1,dims=2)-torch.roll(p,1,dims=2))/(2*self.dy)

		#Central second order difference
		d2udx2 = (torch.roll(u,-1,dims=1)-2*u+torch.roll(u,1,dims=1))/(self.dx**2)
		d2udy2 = (torch.roll(u,-1,dims=2)-2*u+torch.roll(u,1,dims=2))/(self.dy**2)
		d2vdx2 = (torch.roll(v,-1,dims=1)-2*v+torch.roll(v,1,dims=1))/(self.dx**2)
		d2vdy2 = (torch.roll(v,-1,dims=2)-2*v+torch.roll(v,1,dims=2))/(self.dy**2)

		#Bordes are not computed, as it's finite differences makes no sense considering torch.roll method
		u = u[1:-1,1:-1,1:-1]
		v = v[1:-1,1:-1,1:-1]
		p = p[1:-1,1:-1,1:-1]
		dudt = dudt[1:-1,1:-1,1:-1]
		dudx = dudx[1:-1,1:-1,1:-1]
		dudy = dudy[1:-1,1:-1,1:-1]
		dvdt = dvdt[1:-1,1:-1,1:-1]
		dvdx = dvdx[1:-1,1:-1,1:-1]
		dvdy = dvdy[1:-1,1:-1,1:-1]
		dpdx = dpdx[1:-1,1:-1,1:-1]
		dpdy = dpdy[1:-1,1:-1,1:-1]
		d2udx2 = d2udx2[1:-1,1:-1,1:-1]
		d2udy2 = d2udy2[1:-1,1:-1,1:-1]
		d2vdx2 = d2vdx2[1:-1,1:-1,1:-1]
		d2vdy2 = d2vdy2[1:-1,1:-1,1:-1]

		#Calculate residuals
		#Continuity eq
		r0 = torch.abs(dudx+dvdy)
		#Momentum
		r1 = torch.abs(self.rho*(dudt+u*dudx+v*dudy) + dpdx - self.mu*(d2udx2+d2udy2))
		r2 = torch.abs(self.rho*(dvdt+u*dvdx+v*dvdy) + dpdy - self.mu*(d2vdx2+d2vdy2))
		
		return r0, r1, r2
	def forward(self, x, y, std):
		r0_y, r1_y, r2_y = self.calculate_res(y,std)
		r0_x, r1_x, r2_x = self.calculate_res(x,std)

		dif_res_0 = torch.mean((r0_y-r0_x)**2)
		dif_res_1 = torch.mean((r1_y-r1_x)**2)
		dif_res_2 = torch.mean((r2_y-r2_x)**2)

		return 0.001*(dif_res_0+dif_res_1+dif_res_2)
