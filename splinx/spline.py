from jax import jit, random
import jax.numpy as jnp
from functools import partial
import matplotlib.pyplot as plt
from jax import vmap, lax, debug
from jax import custom_jvp

# x shape: (size, x); grid shape: (size, grid)
def extend_grid(grid, k_extend=0, clamp=False):
    # pad k to left and right
    # grid shape: (batch, grid)
    if clamp:
        repeated_first = jnp.repeat(grid[:, :1], k_extend, axis=1)
        repeated_last = jnp.repeat(grid[:, -1:], k_extend, axis=1)
        grid = jnp.concatenate([repeated_first, grid, repeated_last], axis=1)
    else:
        h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

        for i in range(k_extend):
            grid = jnp.concatenate([grid[:, [0]] - h, grid], axis=1)
            grid = jnp.concatenate([grid, grid[:, [-1]] + h], axis=1)

    return grid

@partial(jit, static_argnames=["k"])
def B_batch(x, grid, k=0):
    '''
    evaludate x on B-spline bases
    
    Args:
    -----
        x : 2D jax.numpy.array
            inputs, shape (number of splines, number of samples)
        grid : 2D jax.numpy.array
            grids, shape (number of splines, number of grid points)
        k : int
            the piecewise polynomial order of splines.
    
    Returns:
    --------
        spline values : 3D jax.numpy.array
            shape (number of splines, number of B-spline bases (coeffcients), number of samples). The numbef of B-spline bases = number of grid points + k - 1.
      
    Example
    -------
    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10
    >>> k=3
    >>> key = random.PRNGKey(0)
    >>> x = random.normal(key, shape=(num_spline, num_sample))
    >>> grids = jnp.einsum('i,j->ij', jnp.ones(num_spline), jnp.linspace(-1, 1, num_grid_interval+1))
    >>> extended_grid = extend_grid_jax(grids, k)
    >>> batch_jax = B_batch_jax(x, extended_grid, k=k)
    >>> batch_jax.shape
    (5, 13, 100)
    '''

    value = (x[:, None, :] >= grid[:, :-1, None]) * (x[:, None, :] < grid[:, 1:, None])
    for i in range(1, k + 1):

        denominator1 = grid[:, i:-1, None] - grid[:, :-(i + 1), None]
        denominator2 = grid[:, i + 1:, None] - grid[:, 1:(-i), None]

        condition1 = denominator1 != 0
        condition2 = denominator2 != 0

        value_left = jnp.where(condition1, (x[:, None, :] - grid[:, :-(i + 1), None]) / denominator1 * value[:, :-1], 0)
        value_right = jnp.where(condition2, (grid[:, i + 1:, None] - x[:, None, :]) / denominator2 * value[:, 1:], 0)

        value = value_left + value_right
    
    return value

# @custom_jvp
@partial(jit, static_argnames=["k"])
def coef2curve(x_eval, grid, coef, k):
    '''
    converting B-spline coefficients to B-spline curves. Evaluate x on B-spline curves (summing up B_batch results over B-spline basis).
    
    Args:
    -----
        x_eval : 2D jax.numpy.array
            shape (number of splines, number of samples)
        grid : 2D jax.numpy.array
            shape (number of splines, number of grid points)
        coef : 2D jax.numpy.array
            shape (number of splines, number of coef params). number of coef params = number of grid intervals + k
        k : int
            the piecewise polynomial order of splines.
        device : str
            devicde
        
    Returns:
    --------
        y_eval : 2D jax.numpy.array
            shape (number of splines, number of samples)
        
    Example
    -------
    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10FigureCanvasAgg is non-interactive, and thus cannot be shown
    >>> k = 3
    >>> key = random.PRNGKey(0)
    >>> key, subkey = random.split(key)
    >>> x_eval = random.normal(key, shape=(num_spline, num_sample))
    >>> grids = jnp.einsum('i,j->ij', jnp.ones(num_spline,), jnp.linspace(-1,1,num_grid_interval+1))
    >>> coef = random.normal(subkey, size=(num_spline, num_grid_interval+k))
    >>> extended_grids = extend_grid(grids, k)
    >>> coef2curve(x_eval, extended_grids, coef, k=k).shape
    (5, 100)
    '''
    # x_eval: (size, batch), grid: (size, grid), coef: (size, coef)
    # coef: (size, coef), B_batch: (size, coef, batch), summer over coef
    y_eval = jnp.einsum('ij,ijk->ik', coef, B_batch(x_eval, grid, k))
    return y_eval

@partial(jit, static_argnames=["k"])
def curve2coef(x_eval, y_eval, grid, k):
    '''
    converting B-spline curves to B-spline coefficients using least squares.
    
    Args:
    -----
        x_eval : 2D jax.numpy.array
            shape (number of splines, number of samples)
        y_eval : 2D jax.numpy.array
            shape (number of splines, number of samples)
        grid : 2D jax.numpy.array
            shape (number of splines, number of grid points)
        k : int
            the piecewise polynomial order of splines
        
    Example
    -------
    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10
    >>> k = 3
    >>> x_eval = random.normal(0,1,size=(num_spline, num_sample))
    >>> y_eval = random.normal(0,1,size=(num_spline, num_sample))
    >>> grids = jnp.einsum('i,j->ij', jnp.ones(num_spline,), jnp.linspace(-1,1,num_grid_interval+1))
    >>> extended_grids = extend_grid(grids, k)
    >>> curve2coef(x_eval, y_eval, grids, k=k).shape
    (5, 13)
    '''

    def batched_lstsq(mat, y_eval):
        # Solve the least squares problem, over the batch
        coef, _, _, _ = jnp.linalg.lstsq(mat, y_eval, rcond=None)
        return coef
        
    mat_batch = jnp.transpose(B_batch(x_eval, grid, k), (0, 2, 1))
    vmap_process_batch = vmap(batched_lstsq, in_axes=(0, 0))
    coef_batch = vmap_process_batch(mat_batch, y_eval)

    return coef_batch

# Creates a static version of coef2curve that has a certain order of spline
def create_static_coef2curve(k):

    assert k > 0, "k must be greater than 0"
    
    def B_batch(x, grid):
        value = (x[:, None, :] >= grid[:, :-1, None]) * (x[:, None, :] < grid[:, 1:, None])
        for i in range(1, k + 1):

            denominator1 = grid[:, i:-1, None] - grid[:, :-(i + 1), None]
            denominator2 = grid[:, i + 1:, None] - grid[:, 1:(-i), None]

            condition1 = denominator1 != 0
            condition2 = denominator2 != 0

            value_left = jnp.where(condition1, (x[:, None, :] - grid[:, :-(i + 1), None]) / denominator1 * value[:, :-1], 0)
            value_right = jnp.where(condition2, (grid[:, i + 1:, None] - x[:, None, :]) / denominator2 * value[:, 1:], 0)

            value = value_left + value_right
        return value
    
    def B_batch_lower(x, grid):
        value = (x[:, None, :] >= grid[:, :-1, None]) * (x[:, None, :] < grid[:, 1:, None])
        for i in range(1, k):

            denominator1 = grid[:, i:-1, None] - grid[:, :-(i + 1), None]
            denominator2 = grid[:, i + 1:, None] - grid[:, 1:(-i), None]

            condition1 = denominator1 != 0
            condition2 = denominator2 != 0

            value_left = jnp.where(condition1, (x[:, None, :] - grid[:, :-(i + 1), None]) / denominator1 * value[:, :-1], 0)
            value_right = jnp.where(condition2, (grid[:, i + 1:, None] - x[:, None, :]) / denominator2 * value[:, 1:], 0)

            value = value_left + value_right
        return value
    
    def coef2curve_lower(x_eval, grid, coef):
        y_eval = jnp.einsum('ij,ijk->ik', coef, B_batch_lower(x_eval, grid))
        return y_eval

    @custom_jvp
    def coef2curve(x_eval, grid, coef):
        y_eval = jnp.einsum('ij,ijk->ik', coef, B_batch(x_eval, grid))
        return y_eval
    
    def spline_derivative(x_eval, grid, coef):
        # Implement the function to compute the derivative of B-spline basis wrt x_eval here
        numerator = coef[:, 1:] - coef[:, :-1]
        denominator = grid[:, k+1:-1] - grid[:, 1:-(k+1)]
        new_coef = numerator * k / denominator
        
        # 0/0 defaults to 0 as defined here: https://public.vrac.iastate.edu/~oliver/courses/me625/week5b.pdf
        is_zero_over_zero = (numerator == 0.0) & (denominator == 0.0)

        new_coef = jnp.where(is_zero_over_zero & jnp.isnan(new_coef), 0.0, new_coef)

        y_eval = coef2curve_lower(x_eval, grid[:,1:-1], new_coef)

        return y_eval

    @coef2curve.defjvp
    def coef2curve_jvp(primals, tangents):
        x_eval, grid, coef = primals
        x_eval_dot, _, _ = tangents

        y_eval = coef2curve(x_eval, grid, coef)

        y_eval_dot = spline_derivative(x_eval, grid, coef) * x_eval_dot

        return y_eval, y_eval_dot
    
    return jit(coef2curve)

if __name__=="__main__":

    # How to plot a 2D spline with control points
    num_sample = 100
    k = 3
    x_eval = jnp.repeat(jnp.linspace(-1, 1, num_sample)[None, :], 2, axis=0)
    
    coef = jnp.array([[1,2,1,-1,5,3,6],
                      [1,2,3,4,5,10,15]])
    num_grid_interval = coef.shape[1] - k
    grids = jnp.einsum('i,j->ij', jnp.ones(2,), jnp.linspace(-1,1,num_grid_interval+1))
    extended_grids = extend_grid(grids, k)

    # Multiply the coeficients by the different basis functions
    y_eval = coef2curve(x_eval, extended_grids, coef, k=k)

    plt.figure(figsize=(10, 6))
    plt.plot(y_eval[0, :], y_eval[1, :], label='Spline')

    # Extract control points for each spline
    # Here, extended_grids would represent the x positions of control points if it aligns with the spline's domain
    ctrl_points_x = coef[0, :]  # You might need to adjust this based on your extend_grid function
    ctrl_points_y = coef[1, :]
    
    # Plot control points
    plt.scatter(ctrl_points_x, ctrl_points_y, marker='s', s=50, color='red', label="Control Points")  # Red circles for control points
    # Optionally connect control points with lines to better visualize the influence
    plt.plot(ctrl_points_x, ctrl_points_y, 'r--', alpha=0.5)  # Dashed lines connecting control points

    plt.title('2D B-Spline Curve with Control Points')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)
    plt.show()