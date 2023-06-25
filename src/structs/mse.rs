use ndarray::Array2;

/// MSE represents the Mean Squared Error loss function.
pub struct MSE {
    pub dL_dx: Option<Array2<f64>>,
}

impl MSE {
    /// Create a new MSE loss function.
    pub fn new() -> Self {
        Self { dL_dx: None }
    }

    /// Get the output of the MSE loss function.
    fn get_output(&self, x: &Array2<f64>, target: &Array2<f64>) -> f64 {
        // Calculate the error between target and x
        let error = target - x;

        // Calculate the mean squared error
        let mse = error.mapv(|x| x * x).mean();
        mse.expect("Unable to compute loss")
    }

    /// Perform forward pass through the MSE loss function.
    pub fn forward(&mut self, x: &Array2<f64>, target: &Array2<f64>) -> f64 {
        // Calculate dL/dx = 2 * (x - target) / N, where N is the number of elements in x
        let dL_dx = 2.0 * (x - target) / x.len() as f64;

        // Store dL/dx for later use in backward pass
        self.dL_dx = Some(dL_dx);

        // Get the output of the MSE loss function
        self.get_output(x, target)
    }

    /// Perform backward pass through the MSE loss function.
    pub fn backward(&self) -> Array2<f64> {
        // Return dL/dx obtained from the `self.dL_dx` field
        self.dL_dx.clone().expect("Need to call forward() first.")
    }
}
