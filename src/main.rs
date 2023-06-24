use std::collections::VecDeque;

use crabnn::structs::linear::pdf;
use crabnn::structs::mse::MSE;
use crabnn::structs::nn::NN;

use ndarray::{Array1, Array2};

use rand::Rng;

use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(123);

    let mut nn = NN::new(2, 8, 8, 1, &mut rng);
    let mut loss = MSE::new();

    let batch_size = 16;

    let mut learning_rate = 0.013;

    let mut running_loss = VecDeque::with_capacity(1000);

    for i in 0..5000000 {
        let x: Array2<f64> = Array2::from_shape_fn((batch_size, 2), |_| rng.gen_range(-3.0..=3.0));
        let pdf_values: Array1<f64> = x
            .axis_iter(ndarray::Axis(0))
            .map(|row| {
                let (x, y) = (row[0], row[1]);
                pdf(x, y)
            })
            .collect();
        let target = pdf_values
            .into_shape((batch_size, 1))
            .expect("Unable to covert target to correct shape");

        let y = nn.forward(&x);

        let loss_value = loss.forward(&y, &target);

        running_loss.push_back(loss_value);
        if i > 1000 {
            running_loss.pop_front();
        }

        learning_rate *= 0.99999;

        #[allow(non_snake_case)]
        let dL_dy = loss.backward();

        nn.backward(&dL_dy);

        if i % 10000 == 0 {
            let avg_loss: f64 = running_loss.iter().sum::<f64>() / 1000.0;
            println!("Iter {:?}, loss: {:?}", i, avg_loss);
        }

        if i % 500000 == 0 && i > 1 {
            nn.plot(i)?;
        }

        nn.fc1.W = nn.fc1.W - nn.fc1.dL_dW.clone().expect("No gradient registered") * learning_rate;
        nn.fc1.b = nn.fc1.b - nn.fc1.dL_db.clone().expect("No gradient registered") * learning_rate;

        nn.fc2.W = nn.fc2.W - nn.fc2.dL_dW.clone().expect("No gradient registered") * learning_rate;
        nn.fc2.b = nn.fc2.b - nn.fc2.dL_db.clone().expect("No gradient registered") * learning_rate;

        nn.fc3.W = nn.fc3.W - nn.fc3.dL_dW.clone().expect("No gradient registered") * learning_rate;
        nn.fc3.b = nn.fc3.b - nn.fc3.dL_db.clone().expect("No gradient registered") * learning_rate;
    }

    Ok(())
}
