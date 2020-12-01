# Thesis
Thesis project for the MSc Computational Science
Thesis title: "Machine learning-assisted committor prediction based on molecular simulation data"

In case of usage, please acknowledge it's affiliation with the Van't Hoff Institute for Molecular Sciences at the University of Amsterdam. 

For citation use:
"Martin Frassek, Arjun Wadhawan and Peter G. Bolhuis. “Machine learning-assisted committor prediction based on molecular simulation data” In preparation (2020)."

The extended autoencoder model is a tensorflow-based neural network that was constructed to gain insights on the reaction mechanism of methane hydrate nucleation. Due to its general build, the model can be used for other systems as well.

The model preprocesses incoming data, maps it to a latent space and then both reconstructs the input as well as predicts a committor based on the position on the latent space.

## License
Licensed under the [MIT license](LICENSE)