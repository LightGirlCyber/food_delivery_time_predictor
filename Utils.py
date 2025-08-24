import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plt_house_x(x, y, f_wb=None, ax=None):
    """Plot house data with optional prediction line"""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    ax.scatter(x, y, marker='x', c='r', label='Training data')
    ax.set_xlabel('Size (1000 sqft)')
    ax.set_ylabel('Price (1000s of dollars)')
    ax.set_title('Housing Prices')
    
    if f_wb is not None:
        ax.plot(x, f_wb, c='b', label='Prediction')
        ax.legend()
    
    return ax

def plt_contour_wgrad(x, y, hist, w_range, b_range):
    """Plot contour plot with cost function and gradients"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Create meshgrid for contour plot
    w_grid, b_grid = np.meshgrid(w_range, b_range)
    
    # Calculate cost for each point (simplified version)
    # You'll need to replace this with your actual cost function
    z = np.zeros_like(w_grid)
    for i in range(len(w_range)):
        for j in range(len(b_range)):
            w_val, b_val = w_grid[j, i], b_grid[j, i]
            # Compute cost here - this is a placeholder
            predictions = w_val * x + b_val
            cost = np.mean((predictions - y) ** 2) / 2
            z[j, i] = cost
    
    # Create contour plot
    contour = ax.contour(w_grid, b_grid, z, levels=20)
    ax.clabel(contour, inline=True, fontsize=8)
    
    # Plot gradient descent path if provided
    if hist:
        w_hist = [p[0] for p in hist]
        b_hist = [p[1] for p in hist]
        ax.plot(w_hist, b_hist, 'o-', color='orange', markersize=5)
        ax.plot(w_hist[0], b_hist[0], 'ro', markersize=8, label='Start')
        ax.plot(w_hist[-1], b_hist[-1], 'go', markersize=8, label='End')
    
    ax.set_xlabel('w')
    ax.set_ylabel('b')
    ax.set_title('Cost Function Contour')
    ax.legend()
    
    return ax

def plt_divergence(x, y, w_range, b_range):
    """Plot to show divergence in gradient descent"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: Cost vs w
    costs_w = []
    for w in w_range:
        predictions = w * x + 0  # b = 0 for simplicity
        cost = np.mean((predictions - y) ** 2) / 2
        costs_w.append(cost)
    
    ax1.plot(w_range, costs_w)
    ax1.set_xlabel('w')
    ax1.set_ylabel('Cost')
    ax1.set_title('Cost vs w')
    ax1.grid(True)
    
    # Right plot: Cost vs b
    costs_b = []
    for b in b_range:
        predictions = 1 * x + b  # w = 1 for simplicity
        cost = np.mean((predictions - y) ** 2) / 2
        costs_b.append(cost)
    
    ax2.plot(b_range, costs_b)
    ax2.set_xlabel('b')
    ax2.set_ylabel('Cost')
    ax2.set_title('Cost vs b')
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

def plt_gradients(x, y, w, b, alpha=0.01, num_iters=100):
    """Plot gradient vectors and descent path"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Simple gradient descent visualization
    w_hist, b_hist = [w], [b]
    m = len(x)
    
    for i in range(num_iters):
        # Forward pass
        predictions = w * x + b
        
        # Compute cost
        cost = np.mean((predictions - y) ** 2) / 2
        
        # Compute gradients
        dw = np.mean((predictions - y) * x)
        db = np.mean(predictions - y)
        
        # Update parameters
        w = w - alpha * dw
        b = b - alpha * db
        
        w_hist.append(w)
        b_hist.append(b)
        
        # Plot every 10 iterations
        if i % 10 == 0:
            ax.arrow(w_hist[i], b_hist[i], -alpha * dw * 100, -alpha * db * 100,
                    head_width=0.05, head_length=0.05, fc='red', ec='red')
    
    ax.plot(w_hist, b_hist, 'b-', alpha=0.7, label='Descent Path')
    ax.plot(w_hist[0], b_hist[0], 'go', markersize=8, label='Start')
    ax.plot(w_hist[-1], b_hist[-1], 'ro', markersize=8, label='End')
    
    ax.set_xlabel('w')
    ax.set_ylabel('b')
    ax.set_title('Gradient Descent Path')
    ax.legend()
    ax.grid(True)
    
    return ax

# Additional utility functions that might be needed
def compute_cost(x, y, w, b):
    """Compute the cost function for linear regression"""
    m = x.shape[0]
    predictions = w * x + b
    cost = np.sum((predictions - y) ** 2) / (2 * m)
    return cost

def compute_gradient(x, y, w, b):
    """Compute the gradient for linear regression"""
    m = x.shape[0]
    predictions = w * x + b
    dw = np.sum((predictions - y) * x) / m
    db = np.sum(predictions - y) / m
    return dw, db