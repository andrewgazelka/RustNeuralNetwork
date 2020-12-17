use std::io;
use crate::files::mnist::MNIST;

mod neural_network;
mod utils;
mod files;

fn main() -> Result<(), io::Error> {
    let mnist = MNIST::new("data/train-labels", "data/train-images")?;
    for x in mnist.images.iter() {
        println!("{}", x.to_string())
    }
    Ok(())
}
