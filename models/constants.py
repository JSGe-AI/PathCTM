VALID_NEURON_SELECT_TYPES = ['first-last', 'random', 'random-pairing']

VALID_BACKBONE_TYPES = [
    f'resnet{depth}-{i}' for depth in [18, 34, 50, 101, 152] for i in range(1, 5)
] + ['shallow-wide', 'parity_backbone']

VALID_POSITIONAL_EMBEDDING_TYPES = [
    'learnable-fourier', 'multi-learnable-fourier',
    'custom-rotational', 'custom-rotational-1d'
]
