use std::intrinsics::min_align_of;

use crate::matrix::Matrix;
use crate::utils::{sigmoid, SigmoidData};

struct Connection {
    to: usize,
    weight: f64,
}

struct Node {
    input_value: f64,
    sigmoid: f64,
}

struct Column {
    vec: Vec<Node>
}

struct NeuralNetwork {
    columns: Vec<Column>,
    weight_matrices: Vec<Matrix<f64>>,
}

impl NeuralNetwork {

    fn new(&self, node_count: Vec<f64>)
    fn depth(&self) -> usize {
        self.weight_matrices.len()
    }

    fn matrix_at(&self, i: usize) -> &Matrix<f64> {
        self.weight_matrices.get(i).unwrap()
    }

    fn set_inputs(&mut self, depth: usize, inputs: &Vec<f64>) {
        let column = &mut self.columns[depth];
        let column_vec = &mut column.vec;
        for idx in 0..column_vec.len() {
            let input_value = inputs[idx];
            let sigmoid = sigmoid(input_value);
            column_vec[idx] = Node {
                input_value,
                sigmoid,
            }
        }
    }

    pub fn propagate(&mut self, inputs: &Vec<f64>) {
        let first_column = &mut self.columns[0].vec;
        for (i, node) in first_column.iter_mut().enumerate() {
            node.input_value = inputs[i];
        }
        self.propagate_helper();
    }

    fn propagate_helper(&mut self) {
        for depth in 0..(self.depth() - 1) {
            let matrix_on = self.matrix_at(depth);

            // [ node1weight1 node1weight2 node1weight3 node1weight4]
            // [ node2weight1 node2weight2 node2weight3 node2weight4]
            // [ node3weight1 node3weight2 node3weight3 node3weight4]

            // # columns is number of input
            // # rows is number of output
            // 1st row maps to first output

            let columns = matrix_on.columns;
            let mut outputs = vec![0.0; columns];

            // propagate
            for (row_idx, row) in matrix_on.row_iterator().enumerate() {
                let input = input_vector[row_idx];
                for (column_idx, weight) in row.iter().enumerate() {
                    outputs[column_idx] += weight * input;
                }
            }

            self.set_inputs(depth + 1, &outputs)
        }
    }
}
