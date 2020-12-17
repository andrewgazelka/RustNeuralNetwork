use crate::utils::sigmoid;

#[derive(Copy, Clone, Debug)]
pub struct Node {
    pub input: f64,
    pub sigmoid: f64,
}

impl Node {
    pub fn from_input(input: f64) -> Node {
        Node {
            input,
            sigmoid: sigmoid(input)
        }
    }
}

impl Default for Node {
    fn default() -> Self {
        Node {
            input: 0.0,
            sigmoid: 0.5,
        }
    }
}
