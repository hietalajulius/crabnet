use crabnn::structs::linear::pdf;

use crabnn::structs::mse::MSE;
use crabnn::structs::nn::NN;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

use std::collections::VecDeque;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the random number generator
    let mut rng = StdRng::seed_from_u64(1);

    // Create a new neural network with 2 input features, 8 hidden units in each of the two hidden layers, and 1 output feature
    let mut nn = NN::new(2, vec![8, 8], 1, &mut rng);

    // Create a mean squared error (MSE) loss function
    let mut loss = MSE::new();

    // Set the batch size for training
    let batch_size = 16;

    // Set the initial learning rate
    let mut learning_rate = 0.002;

    // Create a buffer to store the running loss for monitoring training progress
    let mut running_loss = VecDeque::with_capacity(1000);

    for i in 0..5000000 {
        // Generate random input samples within the range [-3.0, 3.0]
        let x: Array2<f64> = Array2::from_shape_fn((batch_size, 2), |_| rng.gen_range(-3.0..=3.0));

        // Compute the target values for the input samples using the PDF function
        let pdf_values: Array1<f64> = x
            .axis_iter(ndarray::Axis(0))
            .map(|row| {
                let (x, y) = (row[0], row[1]);
                pdf(x, y)
            })
            .collect();
        let target = pdf_values
            .into_shape((batch_size, 1))
            .expect("Unable to convert target to correct shape");

        // Perform a forward pass through the neural network to compute the predicted output
        let y = nn.forward(&x);

        // Compute the loss value using the MSE loss function
        let loss_value = loss.forward(&y, &target);

        // Update the running loss buffer
        running_loss.push_back(loss_value);
        if i > 1000 {
            running_loss.pop_front();
        }

        // Update the learning rate
        learning_rate *= 0.99999;

        // Compute the gradients of the loss with respect to the network's parameters
        #[allow(non_snake_case)]
        let dL_dy = loss.backward();
        nn.backward(&dL_dy);

        if i % 10000 == 0 {
            // Compute the average loss over the last 1000 iterations
            let avg_loss: f64 = running_loss.iter().sum::<f64>() / 1000.0;
            println!("Iter {:?}, loss: {:?}", i, avg_loss);
        }

        // Update the network's parameters using stochastic gradient descent
        for layer in nn.layers.iter_mut() {
            layer.sgd(learning_rate);
        }

        if i % 500000 == 0 && i > 1 {
            // Plot the learned probability density function (PDF) at certain iterations
            nn.plot(i)?;
        }
    }

    Ok(())
}
