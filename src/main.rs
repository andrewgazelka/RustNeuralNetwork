use crate::neural_network::NeuralNetwork;

mod neural_network;
mod matrix;
mod utils;
mod node;

fn main() {
    let node_counts: Vec<usize> = vec![4, 3, 1];
    let mut nn = NeuralNetwork::new(&node_counts, 1.0);

    let start = vec![4.0, 3.0, 3.0, 3.0];
    nn.propagate(&start);
    println!("outputs {}", nn.outputs());
}
