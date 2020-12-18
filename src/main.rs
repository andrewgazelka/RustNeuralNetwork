use std::io;

use clap::Clap;
use itertools::Itertools;

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
    height: Option<usize>,

    #[clap(short, long)]
    columns: Option<usize>,
}

fn main() -> Result<(), io::Error> {
    let opts: Opts = Opts::parse();

    match opts.subcmd {
        SubCommand::Numbers(n) => gen_numbers(
            n.height.unwrap_or(usize::max_value()),
            n.columns.unwrap_or(3),
        )
    }?;

    Ok(())
}

fn gen_numbers(height: usize, columns: usize) -> Result<(), io::Error> {
    let mnist = MNIST::new("data/train-labels", "data/train-images")?;

    for images in mnist.images.chunks(columns).take(height) {
        let numbers = images.iter().map(|image| image.to_string()).collect_vec();
        let mut line_iterators = Vec::new();

        for number in &numbers {
            line_iterators.push(number.lines());
        }

        let mut first = true;
        loop {
            let mut has_one = false;
            for (i, lines) in line_iterators.iter_mut().enumerate() {
                if first {
                    print!("{}", images[i].label);
                } else {
                    print!(" ");
                }
                if let Some(line) = lines.next() {
                    has_one = true;
                    print!("{}", line);
                    print!("|")
                }
            }
            println!();
            first = false;
            if !has_one {
                break;
            }
        }
    }
    Ok(())
}
