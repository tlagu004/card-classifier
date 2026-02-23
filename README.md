# Playing Card Classifier

A PyTorch-based computer vision pipeline that classifies images of playing cards into 53 categories (52 cards + 1 Joker). This project utilizes transfer learning with an EfficientNet-B0 backbone.



## Overview
This project follows the three-pillar PyTorch paradigm:
1.  **Dataset:** A custom wrapper for `torchvision.datasets.ImageFolder`.
2.  **Model:** An EfficientNet-B0 architecture with a modified classifier head.
3.  **Training:** A manual loop implementing backpropagation, validation tracking, and progress visualization.

## Getting Started

### Prerequisites
* Python 3.10+
* PyTorch 2.0+
* `timm` (PyTorch Image Models)
* `tqdm`, `matplotlib`, `pandas`

## Inference
To predict a single image:Pythonfrom PIL import Image
original_image, image_tensor = preprocess_image("path/to/card.jpg", transform)
probabilities = predict(model, image_tensor, device)

## Performance
The model employs a pre-trained EfficientNet backbone, achieving rapid convergence.MetricResultEpochs5OptimizerAdam ($lr=0.001$)Final Val Loss0.139LicenseThis project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
Special thanks to Rob Mulla for the inspiration and educational content that guided the development of this project.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
