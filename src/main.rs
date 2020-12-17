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

        loop {
            let mut has_one = false;
            for lines in &mut line_iterators {
                if let Some(line) = lines.next() {
                    has_one = true;
                    print!("{}", line);
                }
            }
            println!();
            if !has_one {
                break;
            }
        }

        // let lines  = images.iter().map(|image| {
        //     let x = image.to_string().lines();
        //     x
        // });


        //
        // for a in gp {
        //
        // }


        // .group_by(|(i, str)| i);
        // .group_by(|(i, line)| i);
        // .group_by(|(i, s)| 1);


        // let lines = images.iter().map(|image| image.to_string().lines().co);
    }
    Ok(())
}
