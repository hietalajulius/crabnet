use ndarray::Array2;

pub struct MSE {
    pub dL_dx: Option<Array2<f64>>,
}

impl MSE {
    pub fn new() -> Self {
        Self { dL_dx: None }
    }

    fn get_output(&self, x: &Array2<f64>, target: &Array2<f64>) -> f64 {
        let error = target - x;
        let mse = (error.clone() * error).mean();
        mse.expect("Unable to compute loss")
    }

    pub fn forward(&mut self, x: &Array2<f64>, target: &Array2<f64>) -> f64 {
        let dL_dx = 2.0 * (x - target) / x.len() as f64;
        self.dL_dx = Some(dL_dx);

        self.get_output(x, target)
    }

    pub fn backward(&self) -> Array2<f64> {
        self.dL_dx.clone().expect("Need to call forward() first.")
    }
}
