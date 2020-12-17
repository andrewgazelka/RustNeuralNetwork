use std::fs::File;
use std::io::{BufReader, Cursor, Read};
use std::io;

use byteorder::{BigEndian, ReadBytesExt};

use crate::utils::matrix::Matrix;

struct Image {
    label: u8,
    data: Matrix<bool>,
}

struct MNIST {
    images: Vec<Image>
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

type Pixel = bool;

struct ImagesFile {
    image_count: u32,
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

    let mut data = Vec::with_capacity(total_bytes as usize);
    reader.read_to_end(&mut data)?;
    Ok(ImagesFile { image_count, rows_per_image, columns_per_image, data })
}

// https://docs.rs/byteorder/1.3.4/byteorder/
impl MNIST {
    /** http://yann.lecun.com/exdb/mnist/ */
    fn new(label_file: &str, image_file: &str) -> Result<MNIST, io::Error> {
        let labels = read_label_file(label_file)?;

        let ImagesFile { rows_per_image, columns_per_image, data, .. } =
            read_image_file(image_file)?;

        let image_size = (rows_per_image * columns_per_image) as usize;

        assert_eq!(labels.len() * image_size, data.len());

        let images = data.chunks(image_size)
            .zip(labels.iter())
            .map(|(chunk,  &label)| Image {
                data: Matrix::from_vec(rows_per_image as usize, columns_per_image as usize, chunk.to_vec()),
                label,
            }).collect();

        Ok(MNIST {
            images
        })


    }
}
