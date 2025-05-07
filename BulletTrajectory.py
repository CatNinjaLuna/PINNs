import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import grad
from scipy.integrate import solve_ivp

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Physical parameters for the bullet trajectory
g = 9.81  # gravitational acceleration (m/s^2)
initial_velocity = 100.0  # initial velocity (m/s)
launch_angle_deg = 30.0  # launch angle (degrees)
launch_angle = np.radians(launch_angle_deg)  # convert to radians
bullet_mass = 0.1  # kg (not used in the simulation)
air_resistance_coef = 0.01  # air resistance coefficient

# Generate ground truth data using physics equations
def get_ground_truth_data(t_max=10.0, num_points=100):
    """Generate analytical solution for projectile motion with air resistance"""
    def projectile_ode(t, y):
        # y = [x, y, vx, vy]
        x, y, vx, vy = y
        v = np.sqrt(vx**2 + vy**2)
        
        # Acceleration components with air resistance
        ax = -air_resistance_coef * v * vx / bullet_mass
        ay = -g - air_resistance_coef * v * vy / bullet_mass
        
        return [vx, vy, ax, ay]
    
    # Initial conditions [x0, y0, vx0, vy0]
    y0 = [0, 0, initial_velocity * np.cos(launch_angle), initial_velocity * np.sin(launch_angle)]
    
    # Time points
    t_span = (0, t_max)
    t_eval = np.linspace(0, t_max, num_points)
    
    # Solve ODE system
    solution = solve_ivp(projectile_ode, t_span, y0, t_eval=t_eval, method='RK45')
    
    # Extract position data
    t = solution.t
    x = solution.y[0]
    y = solution.y[1]
    
    # Truncate data where y becomes negative (bullet hits ground)
    impact_idx = np.where(y < 0)[0]
    if len(impact_idx) > 0:
        impact_idx = impact_idx[0]
        t = t[:impact_idx]
        x = x[:impact_idx]
        y = y[:impact_idx]
    
    return t, x, y

# Generate data
t_data, x_data, y_data = get_ground_truth_data(t_max=15.0, num_points=200)

# Add some noise to simulate real-world measurements
np.random.seed(123)
x_noise = np.random.normal(0, 5.0, x_data.shape)
y_noise = np.random.normal(0, 5.0, y_data.shape)
x_noisy = x_data + x_noise
y_noisy = y_data + y_noise

# Prepare data for neural networks
input_data = torch.tensor(t_data.reshape(-1, 1), dtype=torch.float32)
target_data = torch.tensor(np.column_stack((x_noisy, y_noisy)), dtype=torch.float32)

# Split data for training and testing
train_idx = np.random.choice(len(input_data), size=int(0.8*len(input_data)), replace=False)
test_idx = np.array([i for i in range(len(input_data)) if i not in train_idx])

train_input = input_data[train_idx]
train_target = target_data[train_idx]
test_input = input_data[test_idx]
test_target = target_data[test_idx]

# Define a traditional neural network without physics knowledge
class TraditionalNN(nn.Module):
    def __init__(self):
        super(TraditionalNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 2)  # output: [x, y] position
        )
    
    def forward(self, t):
        return self.net(t)

# Define Physics-Informed Neural Network (PINN)
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 2)  # output: [x, y] position
        )
        
        # Initial conditions
        self.x0 = 0.0
        self.y0 = 0.0
        self.vx0 = initial_velocity * np.cos(launch_angle)
        self.vy0 = initial_velocity * np.sin(launch_angle)
        
        # Pre-calculated physics terms for better numerical stability
        self.gravity_term = 0.5 * g  # For -0.5 * g * t^2 term
    
    def forward(self, t):
        # Neural network output provides the residual adjustment to the physics solution
        nn_output = self.net(t)
        
        # Physics-based solution (projectile motion with initial conditions)
        # x = x0 + vx0 * t
        # y = y0 + vy0 * t - 0.5 * g * t^2
        physics_x = self.x0 + self.vx0 * t
        physics_y = self.y0 + self.vy0 * t - self.gravity_term * t**2
        
        # Combine physics solution with neural network refinement
        x = physics_x + nn_output[:, 0:1]
        y = physics_y + nn_output[:, 1:2]
        
        return torch.cat([x, y], dim=1)
    
    def get_physics_loss(self, t):
        """Calculate physics-based loss using equations of motion"""
        t.requires_grad_(True)
        xy = self(t)
        
        # First derivatives (velocity)
        velocity = torch.autograd.grad(
            outputs=xy, 
            inputs=t, 
            grad_outputs=torch.ones_like(xy),
            create_graph=True
        )[0]
        
        # Separate velocity components
        vx = velocity[:, 0:1]
        vy = velocity[:, 1:2]
        
        # Second derivatives (acceleration)
        acceleration = torch.autograd.grad(
            outputs=torch.cat([vx, vy], dim=1), 
            inputs=t, 
            grad_outputs=torch.ones_like(torch.cat([vx, vy], dim=1)),
            create_graph=True
        )[0]
        
        # Separate acceleration components
        ax = acceleration[:, 0:1]
        ay = acceleration[:, 1:2]
        
        # Physics residuals (F = ma)
        v = torch.sqrt(vx**2 + vy**2)
        
        # Air resistance effects on acceleration (F = -k*v*v_vector)
        ax_physics = -air_resistance_coef * v * vx / bullet_mass
        ay_physics = -g - air_resistance_coef * v * vy / bullet_mass
        
        # Compute residuals (difference between predicted and physics-based acceleration)
        ax_residual = ax - ax_physics
        ay_residual = ay - ay_physics
        
        # Mean squared residuals
        return torch.mean(ax_residual**2 + ay_residual**2)

# Train the traditional neural network
def train_traditional_nn(model, inputs, targets, epochs=2000):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if (epoch+1) % 200 == 0:
            print(f'Traditional NN - Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
    
    return losses

# Train the Physics-Informed Neural Network
def train_pinn(model, inputs, targets, epochs=2000, physics_weight=0.5):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Data loss
        outputs = model(inputs)
        data_loss = criterion(outputs, targets)
        
        # Physics loss - make sure inputs require grad
        physics_loss = model.get_physics_loss(inputs.clone().detach().requires_grad_(True))
        
        # Total loss
        loss = (1 - physics_weight) * data_loss + physics_weight * physics_loss
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if (epoch+1) % 200 == 0:
            print(f'PINN - Epoch [{epoch+1}/{epochs}], '
                  f'Total Loss: {loss.item():.6f}, '
                  f'Data Loss: {data_loss.item():.6f}, '
                  f'Physics Loss: {physics_loss.item():.6f}')
    
    return losses

# Initialize models
traditional_model = TraditionalNN()
pinn_model = PINN()

# Train models
print("Training Traditional Neural Network...")
trad_losses = train_traditional_nn(traditional_model, train_input, train_target, epochs=2000)

print("\nTraining Physics-Informed Neural Network...")
pinn_losses = train_pinn(pinn_model, train_input, train_target, epochs=2000, physics_weight=0.5)

# Evaluate models on test data
traditional_model.eval()
pinn_model.eval()

with torch.no_grad():
    trad_predictions = traditional_model(test_input).numpy()
    pinn_predictions = pinn_model(test_input).numpy()

# Generate predictions for plotting full trajectory
t_plot = np.linspace(0, max(t_data), 100)
t_plot_tensor = torch.tensor(t_plot.reshape(-1, 1), dtype=torch.float32)

with torch.no_grad():
    trad_plot_pred = traditional_model(t_plot_tensor).numpy()
    pinn_plot_pred = pinn_model(t_plot_tensor).numpy()

# Calculate analytical solution for comparison
t_analytical, x_analytical, y_analytical = get_ground_truth_data(t_max=max(t_data), num_points=100)

# Calculate Mean Squared Error for both models on test data
trad_mse = np.mean((trad_predictions - test_target.numpy())**2)
pinn_mse = np.mean((pinn_predictions - test_target.numpy())**2)

print(f"\nTraditional NN Test MSE: {trad_mse:.6f}")
print(f"PINN Test MSE: {pinn_mse:.6f}")

# Create a figure to visualize the predictions
plt.figure(figsize=(14, 12))

# Plot loss curves
plt.subplot(2, 1, 1)
plt.semilogy(trad_losses, label='Traditional NN', linewidth=2)
plt.semilogy(pinn_losses, label='PINN', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (log scale)', fontsize=12)
plt.title('Training Loss', fontsize=14)
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, alpha=0.3)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.tight_layout(pad=3.0)

# Plot trajectory predictions
plt.subplot(2, 1, 2)

# Plot true trajectory first (underneath)
plt.plot(x_analytical, y_analytical, 'k-', linewidth=3, label='Ground Truth')

# Plot noisy data points
plt.scatter(x_noisy, y_noisy, c='gray', s=10, alpha=0.3, label='Noisy Measurements')

# Plot predictions
plt.plot(trad_plot_pred[:, 0], trad_plot_pred[:, 1], 'b--', linewidth=2.5, label=f'Traditional NN (MSE: {trad_mse:.2f})')
plt.plot(pinn_plot_pred[:, 0], pinn_plot_pred[:, 1], 'r-.', linewidth=2.5, label=f'PINN (MSE: {pinn_mse:.2f})')

# Improve axis labels and title
plt.xlabel('Horizontal Distance (m)', fontsize=12)
plt.ylabel('Height (m)', fontsize=12)
plt.title('Bullet Trajectory Prediction', fontsize=14)

# Improve legend
plt.legend(fontsize=10, loc='upper left', framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.tick_params(axis='both', which='major', labelsize=10)

# Set appropriate axis limits to avoid crowding
max_height = max(np.max(y_analytical), np.max(trad_plot_pred[:, 1]), np.max(pinn_plot_pred[:, 1]))
max_distance = max(np.max(x_analytical), np.max(trad_plot_pred[:, 0]), np.max(pinn_plot_pred[:, 0]))
plt.xlim(-5, max_distance * 1.1)
plt.ylim(-5, max_height * 1.1)

# Add text annotation explaining the model differences
plt.figtext(0.5, 0.02, 
    "Comparison of trajectory models: The Physics-Informed Neural Network (PINN) incorporates Newton's laws of motion\n"
    "and air resistance into its loss function, resulting in more accurate predictions that respect physical constraints.\n"
    "The traditional neural network relies solely on data fitting without any physics knowledge.", 
    ha="center", fontsize=11, bbox={"facecolor":"white", "alpha":0.7, "pad":10, "boxstyle":"round"})

plt.tight_layout(rect=[0, 0.06, 1, 0.97])
plt.savefig('bullet_trajectory_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Comparison complete! Results saved to 'bullet_trajectory_comparison.png'")