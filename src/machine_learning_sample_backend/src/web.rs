// #![allow(clippy::new_without_default)]

// use ::alloc::{boxed::Box, string::String};

use crate::model::Model;
use crate::state::{build_and_load_model, Backend};

use burn::tensor::Tensor;

// use wasm_bindgen::prelude::*;

/// Data structure that corresponds to JavaScript class.
/// See: https://rustwasm.github.io/wasm-bindgen/contributing/design/exporting-rust-struct.html
// #[wasm_bindgen]
pub struct Mnist {
    model: Model<Backend>,
}

// #[wasm_bindgen]
impl Mnist {
    /// Constructor called by JavaScripts with the new keyword.
    // #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            model: build_and_load_model(),
        }
    }

    /// Returns the inference results.
    ///
    /// This method is called from JavaScript via generated wrapper code by wasm-bindgen.
    ///
    /// # Arguments
    ///
    /// * `input` - A f32 slice of input 28x28 image
    ///
    /// See bindgen support types for passing and returning arrays:
    /// * https://rustwasm.github.io/wasm-bindgen/reference/types/number-slices.html
    /// * https://rustwasm.github.io/wasm-bindgen/reference/types/boxed-number-slices.html
    ///
    pub fn inference(&self, input: &[f32]) -> Result<Box<[f32]>, String> {
        // Reshape from the 1D array to 3d tensor [batch, height, width]
        let input: Tensor<Backend, 3> = Tensor::from_floats(input).reshape([1, 28, 28]);

        // Normalize input: make between [0,1] and make the mean=0 and std=1
        // values mean=0.1307,std=0.3081 were copied from Pytorch Mist Example
        // https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/mnist/main.py#L122

        let input = ((input / 255) - 0.1307) / 0.3081;

        // Run the tensor input through the model
        let output: Tensor<Backend, 2> = self.model.forward(input);

        // Convert the model output into probability distribution using softmax formula
        let output: Tensor<Backend, 2> = output.clone().exp() / output.exp().sum_dim(1);

        // Flatten output tensor with [1, 10] shape into boxed slice of [f32]
        Ok(output.to_data().value.into_boxed_slice())
    }
}
