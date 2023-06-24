use ndarray::Array2;

use rand::rngs::StdRng;
use rand::Rng;

pub fn pdf(x: f64, y: f64) -> f64 {
    const SDX: f64 = 0.1;
    const SDY: f64 = 0.1;
    const A: f64 = 5.0;
    let x = x / 10.0;
    let y = y / 10.0;

    A * (-x * x / 2.0 / SDX / SDX - y * y / 2.0 / SDY / SDY).exp()
}

pub struct LinearLayer {
    pub W: Array2<f64>,
    pub b: Array2<f64>,
    // Gradient of the Loss
    pub dL_dW: Option<Array2<f64>>,
    pub dL_db: Option<Array2<f64>>,
    // Gradient of the output
    pub dy_dW: Option<Array2<f64>>,
}

impl LinearLayer {
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

    pub fn get_output(&self, x: &Array2<f64>, W: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        let dot = W.dot(&x.t());
        let transposed = &dot.t();

        let b_t = &b.t().to_owned();

        transposed + b_t
    }

    pub fn forward(&mut self, x: &Array2<f64>) -> Array2<f64> {
        self.dy_dW = Some(x.clone());
        self.get_output(x, &self.W, &self.b)
    }

    pub fn backward(&mut self, dL_dy: &Array2<f64>) -> Array2<f64> {
        let dy_dW = self.dy_dW.clone().expect("Need to call forward() first.");
        let dL_dW = dL_dy.t().dot(&dy_dW);

        let dy_dx = &self.W;

        let dL_dx = dL_dy.dot(dy_dx);

        let dL_db = dL_dy.t().dot(&Array2::ones((dL_dy.shape()[0], 1)));

        self.dL_dW = Some(dL_dW);
        self.dL_db = Some(dL_db);

        dL_dx
    }
}
