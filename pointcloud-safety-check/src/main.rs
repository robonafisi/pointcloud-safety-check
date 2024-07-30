[dependencies]
burn = "0.1"  
burn-tensor = "0.1" 


use burn::tensor::Tensor;
use burn::nn::{Conv2d, Linear, ReLU, Sequential, Module, CrossEntropyLoss, Flatten};
use burn::optim::{SGD, Optimizer};
use rand::seq::SliceRandom;
use rand::thread_rng;
use ndarray::{Array, Array3};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

// Define a CNN for image data
struct CNN {
    layers: Sequential,
}

impl CNN {
    fn new() -> Self {
        let layers = Sequential::new(vec![
            Box::new(Conv2d::new(1, 32, (3, 3))),
            Box::new(ReLU::new()),
            Box::new(Conv2d::new(32, 64, (3, 3))),
            Box::new(ReLU::new()),
            Box::new(Flatten::new()),
            Box::new(Linear::new(64 * 22 * 22, 128)),
            Box::new(ReLU::new()),
        ]);
        Self { layers }
    }
}

impl Module for CNN {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.layers.forward(x)
    }
}

// Define a simple MLP for point cloud data
struct PointCloudMLP {
    layers: Sequential,
}

impl PointCloudMLP {
    fn new(input_dim: usize) -> Self {
        let layers = Sequential::new(vec![
            Box::new(Linear::new(input_dim, 128)),
            Box::new(ReLU::new()),
            Box::new(Linear::new(128, 64)),
            Box::new(ReLU::new()),
        ]);
        Self { layers }
    }
}

impl Module for PointCloudMLP {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.layers.forward(x)
    }
}

// Define a combined model
struct CombinedModel {
    cnn: CNN,
    mlp: PointCloudMLP,
    final_layer: Linear,
}

impl CombinedModel {
    fn new(input_dim: usize) -> Self {
        let cnn = CNN::new();
        let mlp = PointCloudMLP::new(input_dim);
        let final_layer = Linear::new(128 + 64, 2); // Combining CNN and MLP outputs

        Self {
            cnn,
            mlp,
            final_layer,
        }
    }
}

impl Module for CombinedModel {
    fn forward(&self, img: &Tensor, pc: &Tensor) -> Tensor {
        let img_features = self.cnn.forward(img);
        let pc_features = self.mlp.forward(pc);
        let combined = Tensor::cat(&[img_features, pc_features], 1);
        self.final_layer.forward(&combined)
    }
}

