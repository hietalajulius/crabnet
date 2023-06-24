use ndarray::Array2;

pub struct ReLU {
    pub dy_dx: Option<Array2<f64>>,
}

impl ReLU {
    pub fn new() -> Self {
        ReLU { dy_dx: None }
    }

    pub fn get_output(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|val| f64::max(val, 0.0))
    }

    pub fn forward(&mut self, x: &Array2<f64>) -> Array2<f64> {
        let dy_dx = x.mapv(|val| if val > 0.0 { 1.0 } else { 0.0 });
        self.dy_dx = Some(dy_dx);
        self.get_output(x)
    }

    pub fn backward(&self, dL_dy: &Array2<f64>) -> Array2<f64> {
        dL_dy * self.dy_dx.clone().expect("Need to call forward() first.")
    }
}
