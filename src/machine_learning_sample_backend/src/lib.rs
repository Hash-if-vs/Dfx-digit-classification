
use web::Mnist;

pub mod model;
pub mod state;
pub mod web;

// extern crate alloc;
// use std::alloc::{boxed::Box, format, string::String};

#[ic_cdk::query]
fn mnist_inference(hand_drawn_digit: Vec<f32>) -> Vec<f32> {
    Mnist::new().inference(&hand_drawn_digit).unwrap().to_vec()
}
