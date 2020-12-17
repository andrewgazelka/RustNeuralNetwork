use crate::neural_network::NeuralNetwork;

mod neural_network;
mod utils;

fn main() {
    let node_counts: Vec<usize> = vec![1, 1, 1];

    let mut nn = NeuralNetwork::new(&node_counts, 1.0);

    let alpha = 0.5;

    let start = vec![4.0];

    println!("outputs {}", nn.output_string());
    println!("weights {}", nn.weight_string());

    nn.propagate(&start);



    let expected = vec![1.0];

    for _ in 0..10000 {
        nn.update_weights(alpha, &expected);
        nn.propagate(&start);
    }

    println!("weights {}", nn.weight_string());

    println!("outputs {}", nn.output_string());
}
