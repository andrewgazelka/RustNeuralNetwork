use crate::matrix::Matrix;
use crate::node::Node;

pub struct NeuralNetwork {
    columns: Vec<Vec<Node>>,
    weight_matrices: Vec<Matrix<f64>>,
}

impl NeuralNetwork {
    pub fn outputs(&self) -> String {
        let mut s = String::new();
        s.push('\n');
        let mut depth = 0;

        loop {
            let mut empty = true;
            for column_idx in 0..self.columns.len() {
                let column = &self.columns[column_idx];
                if let Some(node) = column.get(depth) {
                    empty = false;
                    s.push_str(&node.sigmoid.to_string());
                    s.push_str("    ");
                } else {
                    s.push(' ')
                }
            }
            if empty {
                break;
            }
            depth += 1;
            s.push('\n');
        }
        s
    }
}


impl NeuralNetwork {
    pub fn new(node_counts: &[usize], init: f64) -> NeuralNetwork {
        let mut columns = Vec::with_capacity(node_counts.len());
        let mut weight_matrices = Vec::with_capacity(node_counts.len() - 1);

        for i in 0..(node_counts.len() - 1) {
            let on_count = node_counts[i];
            let next_count = node_counts[i + 1];

            columns.push(vec![Node::default(); on_count]);
            weight_matrices.push(Matrix::new(on_count, next_count, init));
        }

        columns.push(vec![Node::default(); *node_counts.last().unwrap()]);

        NeuralNetwork { columns, weight_matrices }
    }

    fn depth(&self) -> usize {
        self.columns.len()
    }

    fn matrix_at(&self, i: usize) -> &Matrix<f64> {
        self.weight_matrices.get(i).unwrap()
    }

    fn set_inputs(&mut self, depth: usize, inputs: &[f64]) {
        let column = &mut self.columns[depth];
        for idx in 0..column.len() {
            let input = inputs[idx];
            column[idx] = Node::from_input(input);
        }
    }

    pub fn get_outputs(&self) -> Vec<f64> {
        let last = self.columns.last().unwrap();
        last.iter().map(|x| x.sigmoid).collect()
    }

    pub fn propagate(&mut self, inputs: &[f64]) {
        let first_column = &mut self.columns[0];
        for i in 0..first_column.len() {
            first_column[i] = Node::from_input(inputs[i]);
        }
        self.propagate_helper();
    }

    fn propagate_helper(&mut self) {
        for depth in 0..(self.depth() - 1) {
            let transfer_matrix = self.matrix_at(depth);

            // [ node1weight1 node1weight2 node1weight3 node1weight4]
            // [ node2weight1 node2weight2 node2weight3 node2weight4]
            // [ node3weight1 node3weight2 node3weight3 node3weight4]

            // # columns is number of input
            // # rows is number of output
            // 1st row maps to first output

            let columns = transfer_matrix.columns;
            let mut outputs = vec![0.0; columns];

            let input_vector = &self.columns[depth];

            // propagate
            for (row_idx, row) in transfer_matrix.row_iterator().enumerate() {
                let input = &input_vector[row_idx];
                for (column_idx, weight) in row.iter().enumerate() {
                    outputs[column_idx] += weight * input.sigmoid;
                }
            }

            self.set_inputs(depth + 1, &outputs)
        }
    }
}
