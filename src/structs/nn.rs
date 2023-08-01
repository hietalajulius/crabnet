use ndarray::Array2;
use plotters::prelude::{BitMapBackend, ChartBuilder, IntoDrawingArea};
use plotters::series::SurfaceSeries;
use plotters::style::{Color, HSLColor, BLACK, WHITE};
use rand::rngs::StdRng;

use crate::crabnet::CrabNetLayer;

use super::{linear::LinearLayer, relu::ReLU};

pub struct NN {
    pub layers: Vec<Box<dyn CrabNetLayer>>,
}

impl NN {
    pub fn new(
        in_features: usize,
        hidden_sizes: Vec<usize>,
        out_features: usize,
        rng: &mut StdRng,
    ) -> Self {
        // Initialize the neural network with the given layer sizes
        let mut layers: Vec<Box<dyn CrabNetLayer>> = vec![];
        let mut input_size = in_features;
        for output_size in hidden_sizes {
            layers.push(Box::new(LinearLayer::new(input_size, output_size, rng)));
            layers.push(Box::new(ReLU::new()));
            input_size = output_size
        }

        layers.push(Box::new(LinearLayer::new(input_size, out_features, rng)));

        NN { layers }
    }

    pub fn get_output(&self, x: &Array2<f64>) -> Array2<f64> {
        // Forward pass through the neural network to compute the output
        let mut x = x.clone();
        for layer in &self.layers {
            x = layer.get_output(&x);
        }
        x
    }

    pub fn forward(&mut self, x: &Array2<f64>) -> Array2<f64> {
        // Forward pass through the neural network for training
        let mut x = x.clone();
        for layer in self.layers.iter_mut() {
            x = layer.forward(&x);
        }
        x
    }

    pub fn backward(&mut self, dy: &Array2<f64>) -> Array2<f64> {
        // Backward pass through the neural network for training
        let mut dy = dy.clone();
        for layer in self.layers.iter_mut().rev() {
            dy = layer.backward(&dy);
        }
        dy
    }

    pub fn plot(&self, iter: i32) -> Result<(), Box<dyn std::error::Error>> {
        let file_name = format!("./iter_{}.gif", iter);
        let root = BitMapBackend::gif(file_name, (600, 400), 1)?.into_drawing_area();

        for pitch in 0..60 {
            root.fill(&WHITE)?;

            let mut chart = ChartBuilder::on(&root)
                .caption(
                    format!("Learned PDF, iteration {:?}", iter),
                    ("sans-serif", 20),
                )
                .build_cartesian_3d(-3.0..3.0, 0.0..6.0, -3.0..3.0)?;
            chart.with_projection(|mut p| {
                p.pitch = 1.57 - (1.57 - pitch as f64 / 50.0).abs();
                p.scale = 0.7;
                p.into_matrix()
            });

            chart
                .configure_axes()
                .light_grid_style(BLACK.mix(0.15))
                .max_light_lines(3)
                .draw()?;

            chart.draw_series(
                SurfaceSeries::xoz(
                    (-15..=15).map(|x| x as f64 / 5.0),
                    (-15..=15).map(|x| x as f64 / 5.0),
                    |x: f64, y: f64| {
                        self.get_output(
                            &Array2::from_shape_vec((1, 2), vec![x, y])
                                .expect("Unable to convert input to correct shape"),
                        )[[0, 0]]
                    },
                )
                .style_func(&|&v| {
                    (&HSLColor(240.0 / 360.0 - 240.0 / 360.0 * v / 5.0, 1.0, 0.7)).into()
                }),
            )?;
            root.present()?;
        }
        Ok(())
    }
}
