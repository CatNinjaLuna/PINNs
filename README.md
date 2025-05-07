# Bullet Trajectory: Traditional NN vs Physics-Informed Neural Network (PINN)

This repository demonstrates the application of Physics-Informed Neural Networks (PINNs) to model ballistic trajectories, comparing their performance with traditional neural networks.

## Overview

This code implements two different approaches to modeling the trajectory of a bullet:

1. **Traditional Neural Network**: A standard feedforward network trained solely on noisy measurement data without any physics knowledge.
2. **Physics-Informed Neural Network (PINN)**: A neural network that incorporates Newton's laws of motion and air resistance physics into both its architecture and loss function.

The implementation demonstrates how embedding physical laws into neural networks can potentially improve prediction accuracy for physical systems.

## Requirements

-  Python 3.6+
-  PyTorch
-  NumPy
-  SciPy
-  Matplotlib

## Usage

Run the script with:

```bash
python BulletTrajectory.py
```

This will:

1. Generate synthetic trajectory data based on physics equations
2. Add noise to simulate real-world measurements
3. Train both traditional and physics-informed neural networks
4. Compare their performance
5. Generate a visualization saved as `bullet_trajectory_comparison.png`

## Implementation Details

### Data Generation

-  Simulates a bullet trajectory with air resistance using ODE solver
-  Adds Gaussian noise to emulate measurement error
-  Splits data into training and testing sets

### Traditional Neural Network

-  Feedforward network with 4 hidden layers
-  Trained to minimize MSE between predictions and noisy measurements
-  No knowledge of underlying physics

### Physics-Informed Neural Network (PINN)

-  Similar network architecture but with physics-based enhancements:
   -  Direct embedding of initial conditions
   -  Incorporates a physics-based solution as a starting point
   -  Uses automatic differentiation to compute velocities and accelerations
   -  Loss function penalizes violations of physics laws (Newton's laws with air resistance)

## Results Analysis

Based on the output logs and visualization, we can observe:

### Training Performance

-  The traditional neural network shows smooth loss convergence, reducing from 78.73 to 20.18 over 2000 epochs.
-  The PINN shows issues with the physics loss component (NaN values), while the data loss component decreases from 5802.68 to 877.76.

### Test Performance

-  Traditional NN achieved a test MSE of 49.33
-  PINN resulted in a higher test MSE of 551.78

### Visual Analysis

The visualization reveals:

1. The traditional neural network fits the noisy data well but shows some unrealistic trajectory behavior
2. The PINN produces a smoother trajectory that better captures the parabolic nature of projectile motion
3. Despite its higher MSE against the noisy test data, the PINN's trajectory appears more physically plausible, especially in areas with sparse data points

### Issues Identified

The NaN values in the physics loss suggest numerical instabilities in the gradient calculations. This could be addressed by:

-  Adjusting the scaling of the physics loss component
-  Implementing gradient clipping
-  Modifying the architecture of the physics-based component
-  Using more stable numerical differentiation techniques

## Conclusion

This implementation demonstrates the concept of physics-informed neural networks, highlighting both their potential advantages and implementation challenges. While the traditional neural network achieved better data-fitting performance, the PINN approach shows promise for producing more physically realistic predictions, especially in regions with limited training data.

The numerical instabilities (NaN values) in the physics loss highlight the challenges in implementing PINNs and suggest areas for future improvement.

## Future Work

1. Resolve the numerical instability issues in the physics loss calculation
2. Experiment with different network architectures and hyperparameters
3. Test on real-world trajectory data
4. Implement alternative PINN formulations with improved stability
5. Explore other physical systems where PINNs could be beneficial
