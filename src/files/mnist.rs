use std::fs::File;
use std::io::{BufReader, Read};
use std::io;

use byteorder::{BigEndian, ReadBytesExt};

use crate::utils::matrix::Matrix;

pub type Pixel = bool;

pub struct Image {
    pub label: u8,
    pub data: Matrix<bool>,
}

impl ToString for Image {
    fn to_string(&self) -> String {
        let mut string = String::new();
        string.reserve(self.data.size() * 2);
        for row in self.data.row_iterator() {
            for &elem in row {
                let to_add = if elem {
                    "█"
                } else {
                    " "
                };
                string.push_str(to_add)
            }
            string.push('\n');
        }
        string
    }
}

/// <a href="http://yann.lecun.com/exdb/mnist/">MNIST</a> image data set
pub struct MNIST {
    pub images: Vec<Image>
}

fn read_label_file(name: &str) -> Result<Vec<u8>, io::Error> {
    let f = File::open(name)?;
    let mut reader = BufReader::new(f);
    let magic = reader.read_u32::<BigEndian>()?;

    assert_eq!(magic, 2049);
    let item_count = reader.read_u32::<BigEndian>()?;

    let mut vec = Vec::with_capacity(item_count as usize);
    reader.read_to_end(&mut vec)?;
    Ok(vec)
}


struct ImagesFile {
    rows_per_image: u32,
    columns_per_image: u32,
    data: Vec<Pixel>,
}

fn read_image_file(name: &str) -> Result<ImagesFile, io::Error> {
    let f = File::open(name)?;
    let mut reader = BufReader::new(f);
    let magic = reader.read_u32::<BigEndian>()?;
    assert_eq!(magic, 2051);

    let image_count = reader.read_u32::<BigEndian>()?;

    let rows_per_image = reader.read_u32::<BigEndian>()?;
    let columns_per_image = reader.read_u32::<BigEndian>()?;

    let bytes_per_image = rows_per_image * columns_per_image;

    let total_bytes = bytes_per_image * image_count;

    let mut data_bytes = Vec::with_capacity(total_bytes as usize);
    reader.read_to_end(&mut data_bytes)?;

    let data = data_bytes.into_iter().map(|x| x != 0).collect();

    Ok(ImagesFile { rows_per_image, columns_per_image, data })
}

impl MNIST {
    /// https://docs.rs/byteorder/1.3.4/byteorder/
    pub fn new(label_file: &str, image_file: &str) -> Result<MNIST, io::Error> {
        let labels = read_label_file(label_file)?;

        let ImagesFile { rows_per_image, columns_per_image, data } =
            read_image_file(image_file)?;

        let image_size = (rows_per_image * columns_per_image) as usize;

        assert_eq!(labels.len() * image_size, data.len());

        let images = data.chunks(image_size)
            .zip(labels.iter())
            .map(|(chunk, &label)| Image {
                data: Matrix::from_vec(rows_per_image as usize, columns_per_image as usize, chunk.to_vec()),
                label,
            }).collect();

        Ok(MNIST {
            images
        })
    }
}

#[cfg(test)]
mod tests {
    use std::io::Error;

    use crate::files::mnist::MNIST;

    #[test]
    fn it_works() -> Result<(), Error> {
        let mnist = MNIST::new("data/train-labels", "data/train-images")?;
        for x in mnist.images.iter().take(2) {
            println!("{}", x.to_string())
        }
        Ok(())
    }
}
