**Note: This project is not affiliated with the Rust Foundation([🤦](https://twitter.com/rust_foundation/status/1644132378858729474?s=20)).**

# crabnet 🦀🕸️

## Building a Simple Neural Network in Rust

This project aims to showcase the implementation of a simple neural network using the Rust programming language. The primary goal is to provide a clear understanding of the mathematical concepts behind neural networks through a straightforward example. Check out the blog post for more details: https://www.juliushietala.com/blog/crabnet-tutorial

## Example: Estimating the Probability Density Function (PDF) of a 2D Gaussian

|                                                 Target                                                 |                                                   Learned                                                    |
| :----------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------: |
| ![target](https://github.com/hietalajulius/crabnn/assets/4254623/98a522db-0540-407c-bb39-ccb953891d15) | ![iter_5000000](https://github.com/hietalajulius/crabnn/assets/4254623/d9eaf240-788b-49be-a1bb-84aa50feb6e0) |

In this specific example, we aim to estimate the Probability Density Function (PDF) of a 2D Gaussian distribution using a neural network. The inputs to the neural network are the coordinates `(x, y)` , and the target output is the corresponding 2D Gaussian distribution PDF value at each `(x, y)` coordinate.

The training process involves generating random `(x, y)` coordinates within a certain range and calculating the PDF values for each coordinate. These input-output pairs are used to train the neural network. The network learns to approximate the underlying PDF distribution by adjusting its internal parameters through the process of gradient descent.

## How to Run and Reproduce the Results

To run and reproduce the results, follow the steps below:

1.  Install Rust: Make sure that Rust is installed on your system. If Rust is not yet installed, you can easily install it from the official Rust website: [https://www.rust-lang.org/](https://www.rust-lang.org/) ([may require a credit card](https://twitter.com/rust_foundation/status/1644132378858729474?s=20) 💳 in the future)
2.  Clone the Project
3.  Open a terminal or command prompt and navigate to the project directory.
4.  Execute the following command to build and run the project: `cargo run`

    The script will initiate the training process of the neural network, and you will see relevant information and progress printed in the console.

5.  Adjusting Parameters (Optional): Feel free to experiment with the following parameters within the `main` function of the code:

    - `batch_size`: Controls the number of samples used in each iteration.
    - `learning_rate`: Modulates the rate at which the network learns.
    - `hidden_sizes`: Controls the number and size of linear layers (passed in the constructor of `NN` ).

    By modifying these parameters, you can fine-tune the neural network's behavior and observe different outcomes.

**Please note that this project is not officially affiliated with the Rust Foundation ([🤦](https://twitter.com/rust_foundation/status/1644132378858729474?s=20)). It is an independent effort to showcase the capabilities of Rust in the context of neural network development while focusing on the underlying mathematical concepts.**
