use ndarray::Array2;

use crate::crabnet::CrabNetLayer;

/// ReLU represents the Rectified Linear Unit activation function.
pub struct ReLU {
    pub dy_dx: Option<Array2<f64>>,
}

impl CrabNetLayer for ReLU {
    /// Get the output of the ReLU activation function.
    fn get_output(&self, x: &Array2<f64>) -> Array2<f64> {
        // Apply ReLU function element-wise to input x
        x.mapv(|val| f64::max(val, 0.0))
    }
    /// Perform forward pass through the ReLU activation function.
    fn forward(&mut self, x: &Array2<f64>) -> Array2<f64> {
        // Calculate dy/dx for backpropagation
        let dy_dx = x.mapv(|val| if val > 0.0 { 1.0 } else { 0.0 });

        // Store dy/dx for later use in backward pass
        self.dy_dx = Some(dy_dx);

        // Get the output of the ReLU activation function
        self.get_output(x)
    }

    /// Perform backward pass through the ReLU activation function.
    fn backward(&mut self, dL_dy: &Array2<f64>) -> Array2<f64> {
        // Calculate dL/dx = dL/dy * dy/dx
        dL_dy * &self.dy_dx.clone().expect("Need to call forward() first.")
    }

    fn sgd(&mut self, _learning_rate: f64) {}
}

impl ReLU {
    /// Create a new ReLU activation function.
    pub fn new() -> Self {
        ReLU { dy_dx: None }
    }
}
