use std::fmt::Display;
use std::slice::Iter;

pub struct Matrix<T> {
    data: Vec<T>,
    pub columns: usize,
    pub rows: usize,
}

impl<T: Clone> Matrix<T> {
    /**
    Create an m by n matrix with m rows and n columns
    */
    pub fn new(m: usize, n: usize, init: T) -> Matrix<T> {
        let size = n * m;
        Matrix {
            data: vec![init; size],
            columns: n,
            rows: m,
        }
    }
}

impl <T: Display> Matrix<T>{
    pub fn string_repr(&self) -> String {
        let x: Vec<_> = self.row_iterator().map(|row| {
            let row_str: Vec<_> = row.iter().map(|elem| elem.to_string()).collect();
            row_str.join("  ")
        }).collect();
        x.join("\n")
    }


}

/**
    row-wise matrix
*/
impl<T> Matrix<T> {

    pub fn from_vec(m: usize, n:usize, vector: Vec<T>) -> Matrix<T> {
        Matrix {
            data: vector,
            columns: n,
            rows: m
        }
    }

    pub fn iterator(&self) -> impl Iterator<Item=&T> {
        self.data.iter()
    }

    pub fn size(&self) -> usize {
        self.columns * self.rows
    }

    pub fn idx(&self, m: usize, n: usize) -> usize {
        m * self.columns + n
    }

    pub fn row_iterator(&self) -> impl Iterator<Item=&[T]> {
        self.data.chunks(self.columns)
    }

    pub fn row_iterator_mut(&mut self) -> impl Iterator<Item=&mut [T]> {
        self.data.chunks_mut(self.columns)
    }

    pub fn row_at(&self, row_idx: usize) -> &[T] {
        let from = self.idx(row_idx, 0);
        let to = self.idx(row_idx + 1, 0);
        &self.data[from..to]
    }

    #[allow(dead_code)]
    pub fn row_at_mut(&mut self, row_idx: usize) -> &mut [T] {
        let from = self.idx(row_idx, 0);
        let to = self.idx(row_idx + 1, 0);
        &mut self.data[from..to]
    }

    #[allow(dead_code)]
    pub fn get(&self, m: usize, n: usize) -> Option<&T> {
        self.data.get(self.idx(m, n))
    }

    #[allow(dead_code)]
    pub fn set(&mut self, row: usize, column: usize, value: T) {
        let idx = self.idx(row, column);
        self.data[idx] = value;
    }
}
