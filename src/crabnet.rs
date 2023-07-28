use std::any::Any;

use ndarray::Array2;

pub trait CrabNetLayer: Any {
    fn forward(&mut self, x: &Array2<f64>) -> Array2<f64>;
    fn backward(&mut self, dL_dy: &Array2<f64>) -> Array2<f64>;
    fn get_output(&self, x: &Array2<f64>) -> Array2<f64>;
    fn sgd(&mut self, learning_rate: f64);
}
