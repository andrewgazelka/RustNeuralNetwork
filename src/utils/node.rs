use crate::utils::transfer_functions::sigmoid;

#[derive(Copy, Clone, Debug)]
pub struct Node {
    pub input: f64,
    pub sigmoid: f64,
}

impl Node {
    pub fn from_input(input: f64) -> Node {
        Node {
            input,
            sigmoid: sigmoid(input),
        }
    }

    pub fn d_sigmoid(&self) -> f64 {
        let sig = self.sigmoid;
        sig * (1.0 - sig)
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
