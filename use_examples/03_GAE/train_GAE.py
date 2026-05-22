# Graph Autoencoder Training
#
# Estimated Time:
# - CPU: 4-6 hours
# - GPU: 1-2 hours
# Category: Long
#
# Overview: Train AttentiveFP encoder-decoder to learn fragment embeddings
#
# Prerequisites:
# - GSGE installed (see Installation.md)
# - Vocabulary and corpus built
# - GPU recommended (but CPU works, just slower)
# - 4GB+ RAM if GPU, 8GB+ if CPU
#
# Learning Objectives:
# - Configure GAE architecture
# - Train encoder-decoder on fragment graphs
# - Monitor training metrics
# - Save and load trained checkpoints

from GSGE.graphs.fragment_graph.GAE import *
from GSGE.core_gsge import CoreGSGE
from GSGE import get_use_examples_dir

# Get use_examples directory
examples_dir = get_use_examples_dir()
if examples_dir is None:
    raise RuntimeError("Cannot find use_examples directory. Run from source checkout.")

# Load in data
vocab_path = str(examples_dir / '00_making_vocabs' / 'vocabs' / 'GS_vocab_v5_a')
corpus_path = str(examples_dir / '00_making_vocabs' / 'vocabs' / 'GSGE_corpus_v5')
batch_size = 64

train_loader, val_loader = CoreGSGE.load_and_prepare_data(vocab_path, corpus_path, x_percent=0.2, seed=42, batch_size=batch_size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Models
encoder = AttentiveFP(in_channels=9, hidden_channels=256, out_channels=128, edge_dim=3, num_layers=3, num_timesteps=2)
decoder = GraphDecoder(latent_dim=128, hidden_dim=256)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

# Paths
gae_dir = examples_dir / '03_GAE' / 'v2'
checkpoint_dir = str(gae_dir / 'model_checkpoints')
load_checkpoint_path = None

# Initialize trainer
trainer = GraphAutoencoderTrainer(
    encoder=encoder,
    decoder=decoder,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    checkpoint_dir=checkpoint_dir,
    device=device,
    batch_size=batch_size,
    load_checkpoint_path=load_checkpoint_path
)

# Run training
trainer.train(num_epochs=300, checkpoint_interval=5)