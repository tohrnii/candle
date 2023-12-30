
use std::collections::HashMap;
use candle::{IndexOp, Module, Result, Tensor, D};
use candle_nn::{RmsNorm, VarBuilder};

use candle_transformers::models::with_tracing::{linear, linear_no_bias, Linear};

pub struct CLIPVisionTower {
    is_loaded: bool,
    vision_tower_name: String,
    select_layer: usize,
    select_feature: String,
    config: CLIPVisionConfig,
    image_processor: CLIPImageProcessor,
    vision_tower: CLIPVisionModel,
}

pub struct CLIPVisionModel {
    config: CLIPVisionConfig,
    main_input_name: String,

}
impl Module for CLIPVisionModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        todo!()
    }
}

pub struct CLIPImageProcessor {
    do_resize: bool,
    size: HashMap<String, usize>,
    // resample: todo!(),
    do_center_crop: bool,
    crop_size: HashMap<String, usize>,
    do_rescale: bool,
    rescale_factor: f32,
    do_normalize: bool,
    image_mean: Option<Vec<f32>>,
    image_std: Option<Vec<f32>>,
    do_convert_rgb: bool,
}

impl Default for CLIPImageProcessor {
    fn default() -> Self {
        Self {
            do_resize: true,
            size: HashMap::new(),
            // resample: todo(),
            do_center_crop: true,
            crop_size: HashMap::new(),
            do_rescale: true,
            rescale_factor: 1.0 / 255.0,
            do_normalize: true,
            image_mean: None,
            image_std: None,
            do_convert_rgb: true,
        }
    }
}
impl CLIPImageProcessor {
    fn new(
        do_resize: bool,
        size: HashMap<String, usize>,
        do_center_crop: bool,
        crop_size: HashMap<String, usize>,
        do_rescale: bool,
        rescale_factor: f32,
        do_normalize: bool,
        image_mean: Option<Vec<f32>>,
        image_std: Option<Vec<f32>>,
        do_convert_rgb: bool,
    ) -> Self {
        Self {
            do_resize,
            size: if size.is_empty() {
                HashMap::from([("shortest_edge".to_string(), 224)])
            } else {
                size
            },
            do_center_crop,
            crop_size: if crop_size.is_empty() {
                HashMap::from([("height".to_string(), 224), ("width".to_string(), 224)])
            } else {
                crop_size
            },
            do_rescale,
            rescale_factor,
            do_normalize,
            image_mean,
            image_std,
            do_convert_rgb,
        }
    }
}

pub struct CLIPVisionConfig {

}
