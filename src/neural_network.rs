use crate::matrix::Matrix;
use crate::utils::sigmoid;


#[derive(Copy, Clone, Debug)]
struct Node {
    input_value: f64,
    sigmoid: f64,
}

impl Default for Node {
    fn default() -> Self {
        Node {
            input_value: 0.0,
            sigmoid: 0.5,
        }
    }
}

pub struct NeuralNetwork {
    columns: Vec<Vec<Node>>,
    weight_matrices: Vec<Matrix<f64>>,
}

impl NeuralNetwork {
    pub fn new(node_counts: &Vec<usize>, init: f64) -> NeuralNetwork {
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
        self.weight_matrices.len()
    }

    fn matrix_at(&self, i: usize) -> &Matrix<f64> {
        self.weight_matrices.get(i).unwrap()
    }

    fn set_inputs(&mut self, depth: usize, inputs: &Vec<f64>) {
        let vec = &mut self.columns[depth];
        for idx in 0..vec.len() {
            let input_value = inputs[idx];
            let sigmoid = sigmoid(input_value);
            vec[idx] = Node {
                input_value,
                sigmoid,
            }
        }
    }

    pub fn get_outputs(&self) -> Vec<f64> {
        let last = self.columns.last().unwrap();
        last.iter().map(|x| x.sigmoid).collect()
    }

    pub fn propagate(&mut self, inputs: &[f64]) {
        let first_column = &mut self.columns[0];
        for (i, node) in first_column.iter_mut().enumerate() {
            node.input_value = inputs[i];
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
