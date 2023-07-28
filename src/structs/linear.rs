use ndarray::Array2;
use rand::rngs::StdRng;
use rand::Rng;

use crate::crabnet::CrabNetLayer;

/// Probability density function used in the example.
pub fn pdf(x: f64, y: f64) -> f64 {
    const SDX: f64 = 0.1;
    const SDY: f64 = 0.1;
    const A: f64 = 5.0;
    let x = x / 10.0;
    let y = y / 10.0;

    // Formula: A * exp(-x^2 / (2 * SDX^2) - y^2 / (2 * SDY^2))
    A * (-x * x / 2.0 / SDX / SDX - y * y / 2.0 / SDY / SDY).exp()
}

/// LinearLayer represents a linear layer in a neural network.
pub struct LinearLayer {
    pub W: Array2<f64>,
    pub b: Array2<f64>,
    // Gradient of the loss
    pub dL_dW: Option<Array2<f64>>,
    pub dL_db: Option<Array2<f64>>,
    // Gradient of the output
    pub dy_dW: Option<Array2<f64>>,
}

impl CrabNetLayer for LinearLayer {
    /// Get the output of the linear layer.
    fn get_output(&self, x: &Array2<f64>) -> Array2<f64> {
        // Formula: (W * x^T + b)^T
        (self.W.dot(&x.t()) + self.b.clone()).t().to_owned()
    }

    /// Perform forward pass through the linear layer.
    fn forward(&mut self, x: &Array2<f64>) -> Array2<f64> {
        // Store the input gradient for later use in backward pass
        self.dy_dW = Some(x.to_owned());
        self.get_output(x)
    }

    /// Perform backward pass through the linear layer.
    fn backward(&mut self, dL_dy: &Array2<f64>) -> Array2<f64> {
        // Calculate the gradient of the loss with respect to W
        let dL_dW = dL_dy.t().dot(
            &self
                .dy_dW
                .as_ref()
                .expect("Need to call forward() first.")
                .view(),
        );

        // Calculate the gradient of the loss with respect to b
        let dL_db = dL_dy.t().dot(&Array2::ones((dL_dy.shape()[0], 1)));

        // Store the gradients for later use
        self.dL_dW = Some(dL_dW);
        self.dL_db = Some(dL_db.to_owned());

        // Calculate the gradient of the loss with respect to the input
        

        dL_dy.dot(&self.W)
    }

    fn sgd(&mut self, learning_rate: f64) {
        self.W =
            self.W.clone() - self.dL_dW.clone().expect("No gradient registered") * learning_rate;
        self.b =
            self.b.clone() - self.dL_db.clone().expect("No gradient registered") * learning_rate;
    }
}

impl LinearLayer {
    /// Create a new LinearLayer with random weights and biases.
    pub fn new(in_features: usize, out_features: usize, rng: &mut StdRng) -> Self {
        let W = Array2::from_shape_fn((out_features, in_features), |_| rng.gen::<f64>());
        let b = Array2::from_shape_fn((out_features, 1), |_| rng.gen::<f64>());

        LinearLayer {
            W,
            b,
            dL_dW: None,
            dL_db: None,
            dy_dW: None,
        }
    }
}
