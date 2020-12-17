use std::io;

use clap::Clap;

use crate::files::mnist::MNIST;

mod neural_network;
mod utils;
mod files;

#[derive(Clap)]
#[clap(version = "1.0", author = "Andrew Gazelka <gazel007@umn.edu>")]
struct Opts {
    #[clap(subcommand)]
    subcmd: SubCommand,
}

#[derive(Clap)]
enum SubCommand {
    Numbers(Numbers),
}

#[derive(Clap)]
struct Numbers {
    #[clap(short, long)]
    amount: Option<usize>

}

fn main() -> Result<(), io::Error> {
    let opts: Opts = Opts::parse();
    match opts.subcmd {
        SubCommand::Numbers(n) => gen_numbers(n.amount.unwrap_or(usize::max_value()))
    }?;

    Ok(())
}

fn gen_numbers(amount: usize) -> Result<(), io::Error> {
    let mnist = MNIST::new("data/train-labels", "data/train-images")?;
    for x in mnist.images.iter().take(amount) {
        println!("{}", x.to_string())
    }
    Ok(())
}
