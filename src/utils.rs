use std::intrinsics::expf64;

pub fn sigmoid(input: f64) -> f64 {
    1.0 / (1.0 + input.exp())
}

pub struct SigmoidData {
    pub result: f64,
    pub derivative: f64,
}

impl SigmoidData {
    pub fn from(input: f64) -> SigmoidData {
        let result = sigmoid(input);
        let derivative = result * (1.0 - result);
        SigmoidData {
            result,
            derivative,
        }
    }
}
