use burn::tensor::Tensor;
use burn::nn::{Conv2d, Linear, ReLU, Sequential, Module, CrossEntropyLoss, Flatten};
use burn::optim::{SGD, Optimizer};
use rand::seq::SliceRandom;
use rand::thread_rng;
use ndarray::{Array, Array3};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

// Error handling function for mismatched dimensions
fn assert_same_shape(t1: &Tensor, t2: &Tensor, name1: &str, name2: &str) {
    if t1.shape() != t2.shape() {
        panic!(
            "Shape mismatch: {} has shape {:?}, but {} has shape {:?}",
            name1,
            t1.shape(),
            name2,
            t2.shape()
        );
    }
}

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

struct CombinedModel {
    cnn: CNN,
    mlp: PointCloudMLP,
    final_layer: Linear,
}

impl CombinedModel {
    fn new(input_dim: usize) -> Self {
        let cnn = CNN::new();
        let mlp = PointCloudMLP::new(input_dim);
        let final_layer = Linear::new(128 + 64, 2);

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

fn generate_data(num_samples: usize, img_size: (usize, usize), pc_size: usize) -> (Tensor, Tensor, Tensor) {
    let mut rng = thread_rng();

    let images: Vec<Vec<f64>> = (0..num_samples)
        .map(|_| (0..(img_size.0 * img_size.1)).map(|_| rng.gen_range(0.0..1.0)).collect())
        .collect();
    let images_tensor = Tensor::from(images).reshape(&[num_samples, 1, img_size.0, img_size.1]);

    let point_clouds: Vec<Vec<f64>> = (0..num_samples)
        .map(|_| (0..pc_size).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    let point_clouds_tensor = Tensor::from(point_clouds);

    let labels: Vec<u64> = (0..num_samples)
        .map(|i| if images[i][0] + point_clouds[i][0] > 1.0 { 1 } else { 0 })
        .collect();
    let labels_tensor = Tensor::from(labels);

    (images_tensor, point_clouds_tensor, labels_tensor)
}

fn split_data(inputs: &Tensor, labels: &Tensor, split_ratio: f64) -> ((Tensor, Tensor), (Tensor, Tensor)) {
    assert_same_shape(inputs, labels, "inputs", "labels");

    let num_samples = inputs.shape()[0];
    let mut indices: Vec<usize> = (0..num_samples).collect();
    let mut rng = thread_rng();
    indices.shuffle(&mut rng);

    let split_index = (num_samples as f64 * split_ratio).round() as usize;
    let (train_indices, val_indices) = indices.split_at(split_index);

    let train_inputs = Tensor::from(train_indices.iter().map(|&i| inputs[i].clone()).collect::<Vec<_>>());
    let train_labels = Tensor::from(train_indices.iter().map(|&i| labels[i].clone()).collect::<Vec<_>>());
    let val_inputs = Tensor::from(val_indices.iter().map(|&i| inputs[i].clone()).collect::<Vec<_>>());
    let val_labels = Tensor::from(val_indices.iter().map(|&i| labels[i].clone()).collect::<Vec<_>>());

    ((train_inputs, train_labels), (val_inputs, val_labels))
}

fn train_model(
    model: &mut CombinedModel,
    train_imgs: &Tensor,
    train_pcs: &Tensor,
    train_labels: &Tensor,
    val_imgs: &Tensor,
    val_pcs: &Tensor,
    val_labels: &Tensor,
    epochs: usize,
    learning_rate: f64,
) {
    assert_same_shape(train_imgs, train_labels, "train_imgs", "train_labels");
    assert_same_shape(val_imgs, val_labels, "val_imgs", "val_labels");

    let mut optimizer = SGD::new(learning_rate);

    for epoch in 0..epochs {
        let predictions = model.forward(train_imgs, train_pcs);

        let loss = CrossEntropyLoss::forward(&predictions, train_labels);
        println!("Epoch {}: Training Loss = {:?}", epoch, loss);

        model.zero_grad();
        loss.backward();

        optimizer.step(model);

        let val_predictions = model.forward(val_imgs, val_pcs);
        let val_loss = CrossEntropyLoss::forward(&val_predictions, val_labels);
        let correct_predictions = val_predictions
            .argmax(1)
            .eq(val_labels)
            .sum()
            .to_scalar::<f64>();
        let accuracy = correct_predictions / val_imgs.shape()[0] as f64;

        println!("Epoch {}: Validation Loss = {:?}, Accuracy = {:.2}%", epoch, val_loss, accuracy * 100.0);
    }
}

fn main() {
    let img_size = (28, 28);
    let pc_size = 100;
    let num_samples = 1000;
    let split_ratio = 0.8;
    let epochs = 50;
    let learning_rate = 0.01;

    let (images, point_clouds, labels) = generate_data(num_samples, img_size, pc_size);
    let ((train_imgs, train_labels), (val_imgs, val_labels)) = split_data(&images, &labels, split_ratio);
    let ((train_pcs, _), (val_pcs, _)) = split_data(&point_clouds, &labels, split_ratio);

    let mut model = CombinedModel::new(pc_size);
    train_model(&mut model, &train_imgs, &train_pcs, &train_labels, &val_imgs, &val_pcs, &val_labels, epochs, learning_rate);
}
