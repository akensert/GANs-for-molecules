import tensorflow as tf
import pandas as pd
from rdkit import Chem

from models import GraphWGAN, GraphGenerator, GraphEncoder
from sequences import GraphDataset
from utils import save_mol, check_novelty


data = pd.read_csv('../input/qm9.csv')

seq = GraphDataset(
    X=data.smiles.values,
    batch_size=64,
    max_atoms=9,
    atoms=['C', 'F', 'N', 'O'],
    bonds=['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
)

wgan = GraphWGAN(
    GraphGenerator(
        atom_shape=(seq.max_atoms, seq.atom_dim),
        bond_shape=(seq.bond_dim, seq.max_atoms, seq.max_atoms)
    ),
    GraphEncoder(
        out_shape=1,
        max_atoms=seq.max_atoms,
        atom_dim=seq.atom_dim,
        bond_dim=seq.bond_dim,
        gconv_units=[256, 256, 256, 256],
        dense_units=[1024, 1024],
        gconv_activation='relu',
        dense_activation='relu',
    ),
    latent_dim=64,
    generator_steps=1,
    discriminator_steps=1,
    gp_weight=10.0,
)

wgan.compile(
    optimizer_generator=tf.keras.optimizers.Adam(1e-4),
    optimizer_discriminator=tf.keras.optimizers.Adam(1e-4),
)

wgan.fit(seq, epochs=8)

# # Uncomment to train a VAEWGAN
# vaewgan = GraphVAEWGAN(
#     GraphEncoder(
#         out_shape=(64, 2),
#         max_atoms=seq.max_atoms,
#         atom_dim=seq.atom_dim,
#         bond_dim=seq.bond_dim,
#         gconv_units=[256, 256, 256, 256],
#         dense_units=[1024, 1024],
#         gconv_activation='relu',
#         dense_activation='relu',
#     ),
#     GraphGenerator(
#         atom_shape=(seq.max_atoms, seq.atom_dim),
#         bond_shape=(seq.bond_dim, seq.max_atoms, seq.max_atoms)
#     ),
#     GraphEncoder(
#         out_shape=1,
#         max_atoms=seq.max_atoms,
#         atom_dim=seq.atom_dim,
#         bond_dim=seq.bond_dim,
#         gconv_units=[256, 256, 256, 256],
#         dense_units=[1024, 1024],
#         gconv_activation='relu',
#         dense_activation='relu',
#     ),
#     generator_steps=1,
#     discriminator_steps=1,
#     rec_weight_enc=1.0,
#     rec_weight_gen=1e-2,
#     kl_weight=1e-3,
#     gp_weight=10
# )
#
# vaewgan.compile(
#     optimizer_encoder=tf.keras.optimizers.Adam(1e-4),
#     optimizer_generator=tf.keras.optimizers.Adam(1e-4),
#     optimizer_discriminator=tf.keras.optimizers.Adam(1e-4),
# )
#
#
# vaewgan.fit(seq, epochs=5)


i = 1
while i <= 10:
    A, X = wgan.generate()
    mol = seq.graph_to_mol([A, X])
    try:
        print('---'*20)
        print("Generated mol:", Chem.MolToSmiles(mol))
        print("Is novel?", check_novelty(mol, data))
        save_mol(mol, path=f"../images/generated_mol_{i}.png")
        i += 1
    except:
        pass
