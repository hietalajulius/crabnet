use ndarray::Array2;
use plotters::prelude::{BitMapBackend, ChartBuilder, IntoDrawingArea};
use plotters::series::SurfaceSeries;
use plotters::style::{Color, HSLColor, BLACK, WHITE};
use rand::rngs::StdRng;

use super::{linear::LinearLayer, relu::ReLU};

pub struct NN {
    pub fc1: LinearLayer,
    pub activation_fn1: ReLU,
    pub fc2: LinearLayer,
    pub activation_fn2: ReLU,
    pub fc3: LinearLayer,
}

impl NN {
    pub fn new(
        in_features: usize,
        hidden_size1: usize,
        hidden_size2: usize,
        out_features: usize,
        rng: &mut StdRng,
    ) -> Self {
        // Initialize the neural network with the given layer sizes
        NN {
            fc1: LinearLayer::new(in_features, hidden_size1, rng),
            activation_fn1: ReLU::new(),
            fc2: LinearLayer::new(hidden_size1, hidden_size2, rng),
            activation_fn2: ReLU::new(),
            fc3: LinearLayer::new(hidden_size2, out_features, rng),
        }
    }

    pub fn get_output(&self, x: &Array2<f64>) -> Array2<f64> {
        // Forward pass through the neural network to compute the output
        let x = ReLU::get_output(&LinearLayer::get_output(x, &self.fc1.W, &self.fc1.b));
        let x = ReLU::get_output(&LinearLayer::get_output(&x, &self.fc2.W, &self.fc2.b));

        LinearLayer::get_output(&x, &self.fc3.W, &self.fc3.b)
    }

    pub fn forward(&mut self, x: &Array2<f64>) -> Array2<f64> {
        // Forward pass through the neural network for training
        let x = self.activation_fn1.forward(&self.fc1.forward(x));
        let x = self.activation_fn2.forward(&self.fc2.forward(&x));
        self.fc3.forward(&x)
    }

    pub fn backward(&mut self, dy: &Array2<f64>) -> Array2<f64> {
        // Backward pass through the neural network for training
        let dy = self.fc3.backward(dy);
        let dy = self.activation_fn2.backward(&dy);
        let dy = self.fc2.backward(&dy);
        let dy = self.activation_fn1.backward(&dy);
        self.fc1.backward(&dy)
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
