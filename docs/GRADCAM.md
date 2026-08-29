# Grad-CAM

The active Grad-CAM implementation is in `src/brain_tumor_fusion/visualization/gradcam.py`. It registers forward and full-backward hooks on the last convolution in the selected ResNet block, computes channel weights by spatially averaging gradients, applies ReLU, upsamples the map to the input size, and blends it with the RGB image.

The target layer is selected from the final ResNet block (`conv3` or `conv2` when available). Inference targets the predicted class unless another target is supplied.

Grad-CAM is an attribution visualization, not a segmentation mask, diagnostic proof, or clinical validation. The selected README example is a historical/unverified artifact derived from local data and checkpoints. Raw heatmaps remain ignored because they derive from restricted local medical data.
