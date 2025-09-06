import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR

# Parameters
x_min, x_max = 0, 2     # Spatial domain
t_min, t_max = 0, 0.48  # Time domain
viscosity = 0.01/np.pi  # Viscosity coefficient (for viscous case)

class BurgersPINN(nn.Module):
    def __init__(self, layers=[2, 32, 128, 16, 128, 32, 1]):
        super().__init__()
        
        # Network layers
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            
        # Initialize weights with Xavier
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
        
        self.activation = nn.Tanh()
    
    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        
        for i in range(len(self.layers)-1):
            x = self.activation(self.layers[i](x))
        x = self.layers[-1](x)
        return x

class ViscousBurgersSolver:
    def __init__(self, network, device='cpu'):
        self.network = network.to(device)
        self.device = device
        
        # Generate training points
        self.generate_training_data()
        
        # Move data to device
        self.X_ic = self.X_ic.to(device)
        self.u_ic = self.u_ic.to(device)
        self.X_bc_left = self.X_bc_left.to(device)
        self.X_bc_right = self.X_bc_right.to(device)
        self.X_colloc = self.X_colloc.to(device)
    
    def generate_training_data(self):
        # Initial condition points: u(x,0) = -sin(πx)
        x_ic = torch.linspace(x_min, x_max, 100).reshape(-1, 1)
        t_ic = torch.zeros_like(x_ic)
        self.X_ic = torch.cat((x_ic, t_ic), dim=1)
        self.u_ic = torch.sin(np.pi * x_ic)
        
        # Boundary condition points
        t_bc = torch.linspace(t_min, t_max, 100).reshape(-1, 1)
        x_bc_left = torch.zeros_like(t_bc)
        x_bc_right = torch.ones_like(t_bc) * x_max
        self.X_bc_left = torch.cat((x_bc_left, t_bc), dim=1)
        self.X_bc_right = torch.cat((x_bc_right, t_bc), dim=1)
        
        # Collocation points for PDE
        n_colloc = 10000
        x_colloc = torch.rand(n_colloc, 1) * (x_max - x_min) + x_min
        t_colloc = torch.rand(n_colloc, 1) * (t_max - t_min) + t_min
        self.X_colloc = torch.cat((x_colloc, t_colloc), dim=1)
    
    def compute_pde_residual(self, x_colloc):
        x_colloc.requires_grad = True
        u = self.network(x_colloc)
        
        # Calculate derivatives
        u_grad = torch.autograd.grad(u.sum(), x_colloc, create_graph=True)[0]
        u_t = u_grad[:, 1:2]  # ∂u/∂t
        u_x = u_grad[:, 0:1]  # ∂u/∂x
        
        # Second derivative ∂²u/∂x²
        u_xx = torch.autograd.grad(u_x.sum(), x_colloc, create_graph=True)[0][:, 0:1]
        
        # PDE residual: ∂u/∂t + u*∂u/∂x - ν*∂²u/∂x²
        residual = u_t + u * u_x - viscosity * u_xx
        
        return residual
    
    def loss_ic(self, x, u):
        u_pred = self.network(x)
        return torch.mean((u_pred - u) ** 2)
    
    def loss_bc(self, x):
        u_pred = self.network(x)
        return torch.mean(u_pred ** 2)
    
    def loss_pde(self, x):
        residual = self.compute_pde_residual(x)
        return torch.mean(residual ** 2)
    
    def train(self, epochs=2000, learning_rate=0.001):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        scheduler = ExponentialLR(optimizer, gamma=0.9999)
        
        history = {'total_loss': [], 'ic_loss': [], 'bc_loss': [], 'pde_loss': []}
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Compute losses
            ic_loss = self.loss_ic(self.X_ic, self.u_ic)
            bc_loss = self.loss_bc(self.X_bc_left) + self.loss_bc(self.X_bc_right)
            pde_loss = self.loss_pde(self.X_colloc)
            
            # Total loss
            total_loss = ic_loss + bc_loss + pde_loss
            
            # Backpropagation
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Store losses
            if epoch % 100 == 0:
                history['total_loss'].append(total_loss.item())
                history['ic_loss'].append(ic_loss.item())
                history['bc_loss'].append(bc_loss.item())
                history['pde_loss'].append(pde_loss.item())
                # print all losses
                print(f"Epoch {epoch}: Total Loss = {total_loss.item():.6f}, "
                        f"IC Loss = {ic_loss.item():.6f}, "
                        f"BC Loss = {bc_loss.item():.6f}, "
                        f"PDE Loss = {pde_loss.item():.6f}")
                # print(f"Epoch {epoch}: Loss = {total_loss.item():.6f}")
        
        return history

class InviscidBurgersSolver:
    def __init__(self, network, device='cpu'):
        self.network = network.to(device)
        self.device = device
        
        # Generate training points
        self.generate_training_data()
        
        # Move data to device
        self.X_ic = self.X_ic.to(device)
        self.u_ic = self.u_ic.to(device)
        self.X_bc_left = self.X_bc_left.to(device)
        self.X_bc_right = self.X_bc_right.to(device)
        self.X_colloc = self.X_colloc.to(device)
    
    def generate_training_data(self):
        # Use more points near potential shock formation
        x_ic = torch.linspace(x_min, x_max, 200).reshape(-1, 1)  # More IC points
        t_ic = torch.zeros_like(x_ic)
        self.X_ic = torch.cat((x_ic, t_ic), dim=1)
        self.u_ic = torch.sin(np.pi * x_ic)
        
        # Boundary conditions
        t_bc = torch.linspace(t_min, t_max, 100).reshape(-1, 1)
        x_bc_left = torch.zeros_like(t_bc)
        x_bc_right = torch.ones_like(t_bc) * x_max
        self.X_bc_left = torch.cat((x_bc_left, t_bc), dim=1)
        self.X_bc_right = torch.cat((x_bc_right, t_bc), dim=1)
        
        # More collocation points with adaptive distribution
        n_colloc = 20000  # More collocation points
        x_colloc = torch.rand(n_colloc, 1) * (x_max - x_min) + x_min
        t_colloc = torch.rand(n_colloc, 1) * (t_max - t_min) + t_min
        
        # Add extra points near potential shock formation time
        t_shock = 0.25  # Approximate shock formation time
        extra_points = 5000
        x_extra = torch.rand(extra_points, 1) * (x_max - x_min) + x_min
        t_extra = torch.normal(t_shock, 0.05, (extra_points, 1))
        t_extra = torch.clamp(t_extra, t_min, t_max)
        
        x_colloc = torch.cat([x_colloc, x_extra])
        t_colloc = torch.cat([t_colloc, t_extra])
        self.X_colloc = torch.cat((x_colloc, t_colloc), dim=1)
    
    def compute_pde_residual(self, x_colloc):
        x_colloc.requires_grad = True
        # u should be the result of  t * Network + sin(pi*x)
        # This is to ensure the network predicts the solution at t=0 correctly
        # get the vector of t_colloc from x_colloc
        tc_colloc = x_colloc[:, 1:2]  # Extract time component
        xc_colloc = x_colloc[:, 0:1]  # Extract space component

        u = tc_colloc * (xc_colloc - x_max) * (xc_colloc + x_min) * self.network(x_colloc) + torch.sin(np.pi * xc_colloc)

        # u = self.network(x_colloc)
        
        # Calculate derivatives
        u_grad = torch.autograd.grad(u.sum(), x_colloc, create_graph=True)[0]
        u_t = u_grad[:, 1:2]  # ∂u/∂t
        u_x = u_grad[:, 0:1]  # ∂u/∂x
        
        # Inviscid Burgers equation: ∂u/∂t + u*∂u/∂x = 0
        residual = u_t + u * u_x
        
        # Add artificial viscosity near shocks
        u_xx = torch.autograd.grad(u_x.sum(), x_colloc, create_graph=True)[0][:, 0:1]
        shock_detector = torch.abs(u_x)
        artificial_viscosity = 0.001 * shock_detector * u_xx
        
        return residual - artificial_viscosity
    
    def entropy_loss(self, x):
        """Additional loss term to enforce entropy condition"""
        x.requires_grad = True

        tc_colloc = x[:, 1:2]  # Extract time component
        xc_colloc = x[:, 0:1]  # Extract space component

        u = tc_colloc * (xc_colloc - x_max) * (xc_colloc + x_min)* self.network(x) + torch.sin(np.pi * xc_colloc)

        # u = self.network(x)
        u_grad = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_x = u_grad[:, 0:1]
        
        # Entropy condition: ∂u/∂x should be bounded
        return torch.mean(torch.exp(1.0 * torch.abs(u_x)))
    
    def loss_ic(self, x, u):
        tc_colloc = x[:, 1:2]  # Extract time component
        xc_colloc = x[:, 0:1]  # Extract space component

        u_pred = tc_colloc* (xc_colloc - x_max) * (xc_colloc + x_min) * self.network(x) + torch.sin(np.pi * xc_colloc)
        # u_pred = self.network(x)
        return torch.mean((u_pred - u) ** 2)
    
    def loss_bc(self, x):
        tc_colloc = x[:, 1:2]  # Extract time component
        xc_colloc = x[:, 0:1]  # Extract space component

        u_pred = tc_colloc* (xc_colloc - x_max) * (xc_colloc + x_min) * self.network(x) + torch.sin(np.pi * xc_colloc)
        # u_pred = self.network(x)
        return torch.mean(u_pred ** 2)
    
    def loss_pde(self, x):
        residual = self.compute_pde_residual(x)
        return torch.mean(residual ** 2)
    
    def train(self, epochs=2000, learning_rate=0.001):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        scheduler = ExponentialLR(optimizer, gamma=0.9999)
        
        history = {'total_loss': [], 'ic_loss': [], 'bc_loss': [], 
                  'pde_loss': [], 'entropy_loss': []}
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Compute losses
            ic_loss = self.loss_ic(self.X_ic, self.u_ic)
            bc_loss = self.loss_bc(self.X_bc_left) + self.loss_bc(self.X_bc_right)
            pde_loss = self.loss_pde(self.X_colloc)
            entropy_loss = self.entropy_loss(self.X_colloc)
            
            # Total loss with entropy condition
            total_loss = pde_loss 
            
            # Backpropagation
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Store losses
            if epoch % 100 == 0:
                history['total_loss'].append(total_loss.item())
                history['ic_loss'].append(ic_loss.item())
                history['bc_loss'].append(bc_loss.item())
                history['pde_loss'].append(pde_loss.item())
                history['entropy_loss'].append(entropy_loss.item())
                # print all losses
                print(f"Epoch {epoch}: Total Loss = {total_loss.item():.6f}, "
                      f"IC Loss = {ic_loss.item():.6f}, "
                      f"BC Loss = {bc_loss.item():.6f}, "
                      f"PDE Loss = {pde_loss.item():.6f}, "
                      f"Entropy Loss = {entropy_loss.item():.6f}")
                #print(f"Epoch {epoch}: Loss = {total_loss.item():.6f}")
        
        return history

def predict_solution_inviscid(solver, x, t):
    """Generate predictions for given space-time points."""
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    if not torch.is_tensor(t):
        t = torch.tensor(t, dtype=torch.float32)
    
    x = x.reshape(-1, 1).to(solver.device)
    t = t.reshape(-1, 1).to(solver.device)
    
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
    X_pred = torch.stack([X.flatten(), T.flatten()], dim=1).to(solver.device)
    
    solver.network.eval()
    with torch.no_grad():
        # Extract space and time components from X_pred
        x_pred = X_pred[:, 0:1]
        t_pred = X_pred[:, 1:2]
        # Compute the solution
        u_pred = t_pred * (x_pred - x_max) * (x_pred + x_min) * solver.network(X_pred) + torch.sin(np.pi * x_pred)
    
    return u_pred.reshape(x.shape[0], t.shape[0]).cpu().numpy()

def predict_solution_viscous(solver, x, t):
    """Generate predictions for given space-time points."""
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    if not torch.is_tensor(t):
        t = torch.tensor(t, dtype=torch.float32)
    
    x = x.reshape(-1, 1).to(solver.device)
    t = t.reshape(-1, 1).to(solver.device)
    
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
    X_pred = torch.stack([X.flatten(), T.flatten()], dim=1).to(solver.device)
    
    solver.network.eval()
    with torch.no_grad():
        u_pred = solver.network(X_pred)
    
    return u_pred.reshape(x.shape[0], t.shape[0]).cpu().numpy()

def plot_solutions(x, t, u_viscous, u_inviscid):
    """Plot and compare viscous and inviscid solutions."""
    os.makedirs('./plots-tutorial', exist_ok=True)
    
    # Create time slices plot
    plt.figure(figsize=(15, 5))
    times = [0.0, 0.25, 0.48]
    time_indices = [np.abs(t - time).argmin() for time in times]
    
    for i, (time, idx) in enumerate(zip(times, time_indices)):
        plt.subplot(1, 3, i+1)
        plt.plot(x, u_viscous[:, idx], 'r-', label='Viscous')
        plt.plot(x, u_inviscid[:, idx], 'b--', label='Inviscid')
        plt.xlabel('x')
        plt.ylabel('u')
        plt.title(f't = {time:.2f}')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('./plots-tutorial/comparison_slices.png')
    plt.close()
    
    # Create contour plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    X, T = np.meshgrid(t, x)
    
    c1 = ax1.contourf(T, X, u_viscous, levels=50, cmap='rainbow')
    plt.colorbar(c1, ax=ax1)
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')
    ax1.set_title('Viscous Solution')
    
    c2 = ax2.contourf(T, X, u_inviscid, levels=50, cmap='rainbow')
    plt.colorbar(c2, ax=ax2)
    ax2.set_xlabel('t')
    ax2.set_ylabel('x')
    ax2.set_title('Inviscid Solution')
    
    plt.tight_layout()
    plt.savefig('./plots-tutorial/comparison_contours.png')
    plt.close()

if __name__ == "__main__":
    # Set device and random seeds
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate space-time grid for predictions
    x = np.linspace(x_min, x_max, 100)
    t = np.linspace(t_min, t_max, 100)
    
    # Create and train viscous solver
    network_viscous = BurgersPINN().to(device)
    solver_viscous = ViscousBurgersSolver(network_viscous, device)
    history_viscous = solver_viscous.train()
    
    # Create and train inviscid solver
    network_inviscid = BurgersPINN().to(device)
    solver_inviscid = InviscidBurgersSolver(network_inviscid, device)
    history_inviscid = solver_inviscid.train()
    
    # Generate predictions
    u_viscous = predict_solution_viscous(solver_viscous, x, t)
    u_inviscid = predict_solution_inviscid(solver_inviscid, x, t)
    
    # Plot and compare solutions
    plot_solutions(x, t, u_viscous, u_inviscid)
    
    # Print final errors
    print("\nTraining completed! Final losses:")
    print(f"Viscous solver: {history_viscous['total_loss'][-1]:.6f}")
    print(f"Inviscid solver: {history_inviscid['total_loss'][-1]:.6f}")
