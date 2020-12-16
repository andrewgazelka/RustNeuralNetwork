use std::intrinsics::min_align_of;

pub struct Matrix<T> {
    data: Vec<T>,
    pub columns: usize,
    pub rows: usize,
}

struct MatrixRowIterator<'a, T> {
    row_on: usize,
    matrix: &'a Matrix<T>,
}

impl<'a, T> Iterator for MatrixRowIterator<T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.row_on == self.matrix.rows {
            return None;
        }
        let row = self.matrix.row_at(self.row_on);
        self.row_on += 1;
        Some(row)
    }
}

/**
    row-wise matrix
*/
impl<T> Matrix<T> {
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

    pub fn idx(&self, m: usize, n: usize) -> usize {
        return m * self.columns + n;
    }

    pub fn row_iterator(&self) -> impl Iterator<Item=&[T]> {
        return MatrixRowIterator {
            row_on: 0,
            matrix: self,
        };
    }


    pub fn row_at(&self, row_idx: usize) -> &[T] {
        let from = self.idx(row_idx, 0);
        let to = self.idx(row_idx + 1, 0);
        &self.data[from..to]
    }

    pub fn get(&self, m: usize, n: usize) -> Option<&T> {
        self.data.get(self.idx(m, n))
    }

    pub fn set(&mut self, row: usize, column: usize, value: T) {
        self.data[self.idx(row, column)] = value;
    }
}
