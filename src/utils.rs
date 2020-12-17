pub fn sigmoid(input: f64) -> f64 {
    1.0 / (1.0 + input.exp())
}
