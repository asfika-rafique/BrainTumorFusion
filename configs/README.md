# Configurations

The active configurations are image-only experiments. Paths are repository-relative and are resolved at runtime from the config location.

- `resnet18_image_only.yaml`: baseline ResNet-18 experiment.
- `resnet50_image_only.yaml`: ResNet-50 experiment matching the preserved 0.830-named checkpoint.
- `sanity_image_only.yaml`: short, non-pretrained smoke configuration.
- `clean_resnet18_image_only.yaml`: release-candidate image-only pipeline with exact-duplicate-group splitting and validation-only checkpoint selection.

The old text-fusion configuration is archived because its encoder was a placeholder. The 0.830 filename is a historical label, not a verified performance result.
