## Generative models for molecules (GAN, WGAN, VAEGAN and VAEWGAN)

### Work in progess (unfinished)

### Models:
GAN, WGAN, VAEGAN, VAEWGAN

### Requirements:
* RDKit
* TensorFlow
* NumPy
* Pandas

### Weaknesses of models:
* **[VAEGAN/VAEWGAN]** Reconstruction loss is sub-optimal, not allowing for good reconstruction for the encoder-decoder network
* **[GAN/WGAN]** No additional implementation to tackle mode collapse (e.g., feature matching, mini-batch discrimination, label smoothing etc.)

### Train WGAN to generate molecules
Run from terminal: `python train_and_predict.py`. Generated molecules are outputted to `images/``
