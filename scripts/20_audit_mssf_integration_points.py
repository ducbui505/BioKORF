"""Audit safe future integration points between MSSF and the BioKORF KG encoder."""

from __future__ import annotations

import pickle
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MSSF_PATH = PROJECT_ROOT / "mssf.py"
MODEL_PATH = PROJECT_ROOT / "model.py"
KG_ENCODER_PATH = PROJECT_ROOT / "models" / "kg_encoder.py"
DRUG_SIDE_PATH = PROJECT_ROOT / "Datas" / "drug_side.pkl"
REPORT_PATH = PROJECT_ROOT / "data_processed" / "architecture" / "mssf_integration_audit.txt"


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required audit input not found: {path}")


def line_of(text: str, fragment: str, path: Path) -> int:
    for line_number, line in enumerate(text.splitlines(), start=1):
        if fragment in line:
            return line_number
    raise ValueError(f"Expected code fragment not found in {path}: {fragment!r}")


def line_ref(path: Path, line: int) -> str:
    return f"{path.name}:{line}"


def main() -> None:
    for path in (MSSF_PATH, MODEL_PATH, KG_ENCODER_PATH, DRUG_SIDE_PATH):
        require_file(path)
    mssf = MSSF_PATH.read_text(encoding="utf-8")
    model = MODEL_PATH.read_text(encoding="utf-8")
    kg_encoder = KG_ENCODER_PATH.read_text(encoding="utf-8")

    required_model_fragments = (
        "EncoderConnection(drugs_inputdim=757*11,sides_inputdim=994*4",
        "EncoderAddition(drugs_inputdim=757,sides_inputdim=994",
        "CrossProduction(cross_dim=128,feature_dim=128",
        "features = torch.cat((feature1,feature2,feature3),dim=1)",
        "features = self.attention(features)",
        "mu,logvar = self.gaussian_parametrizer(features)",
        "results = self.classifier(latent_features)",
        "QK = torch.matmul(Q, K.transpose(-1, -2))",
    )
    for fragment in required_model_fragments:
        line_of(model, fragment, MODEL_PATH)
    required_mssf_fragments = (
        "interaction_target[k, 0] = i",
        "interaction_target[k, 1] = j",
        "interaction_target[k, 2] = DAL[i, j]",
        "drug_train = drug_features_matrix[data_train[:, 0]]",
        "side_train = side_features_matrix[data_train[:, 1]]",
        "trainset = torch.utils.data.TensorDataset",
        "batch_drug, batch_side, batch_ratings = data",
        "loss_func = nn.CrossEntropyLoss()",
    )
    for fragment in required_mssf_fragments:
        line_of(mssf, fragment, MSSF_PATH)
    for fragment in ("class BioKORFKGEncoder", "missing_drug_kg_embedding", "missing_side_kg_embedding"):
        line_of(kg_encoder, fragment, KG_ENCODER_PATH)

    with DRUG_SIDE_PATH.open("rb") as handle:
        drug_side = pickle.load(handle)
    if getattr(drug_side, "shape", None) != (757, 994):
        raise ValueError(f"Expected Datas/drug_side.pkl shape (757, 994); found {getattr(drug_side, 'shape', None)}")

    refs = {
        "read_return": line_ref(MSSF_PATH, line_of(mssf, "return drug_features, side_features", MSSF_PATH)),
        "hstack_drug": line_ref(MSSF_PATH, line_of(mssf, "drug_features_matrix = np.hstack", MSSF_PATH)),
        "hstack_side": line_ref(MSSF_PATH, line_of(mssf, "side_features_matrix = np.hstack", MSSF_PATH)),
        "select_train_drug": line_ref(MSSF_PATH, line_of(mssf, "drug_train = drug_features_matrix[data_train[:, 0]]", MSSF_PATH)),
        "select_train_side": line_ref(MSSF_PATH, line_of(mssf, "side_train = side_features_matrix[data_train[:, 1]]", MSSF_PATH)),
        "tensor_dataset": line_ref(MSSF_PATH, line_of(mssf, "trainset = torch.utils.data.TensorDataset", MSSF_PATH)),
        "dataloader": line_ref(MSSF_PATH, line_of(mssf, "_train = torch.utils.data.DataLoader", MSSF_PATH)),
        "batch_loop": line_ref(MSSF_PATH, line_of(mssf, "for i, data in enumerate(train_loader, 0)", MSSF_PATH)),
        "batch_unpack": line_ref(MSSF_PATH, line_of(mssf, "batch_drug, batch_side, batch_ratings = data", MSSF_PATH)),
        "sample_alloc": line_ref(MSSF_PATH, line_of(mssf, "interaction_target = np.zeros", MSSF_PATH)),
        "sample_drug": line_ref(MSSF_PATH, line_of(mssf, "interaction_target[k, 0] = i", MSSF_PATH)),
        "sample_side": line_ref(MSSF_PATH, line_of(mssf, "interaction_target[k, 1] = j", MSSF_PATH)),
        "sample_label": line_ref(MSSF_PATH, line_of(mssf, "interaction_target[k, 2] = DAL[i, j]", MSSF_PATH)),
        "loss": line_ref(MSSF_PATH, line_of(mssf, "Loss = multi_loss+0.001*kl_div", MSSF_PATH)),
        "attention_q": line_ref(MODEL_PATH, line_of(model, "Q = self.WQ(x).view", MODEL_PATH)),
        "attention_score": line_ref(MODEL_PATH, line_of(model, "QK = torch.matmul(Q, K.transpose", MODEL_PATH)),
        "en_con": line_ref(MODEL_PATH, line_of(model, "self.encoderConnection = EncoderConnection", MODEL_PATH)),
        "en_add": line_ref(MODEL_PATH, line_of(model, "self.encoderAddition = EncoderAddition", MODEL_PATH)),
        "cnn": line_ref(MODEL_PATH, line_of(model, "self.crossProduction = CrossProduction", MODEL_PATH)),
        "final_attention": line_ref(MODEL_PATH, line_of(model, "features = self.attention(features)", MODEL_PATH)),
        "feature_concat": line_ref(MODEL_PATH, line_of(model, "features = torch.cat((feature1,feature2,feature3),dim=1)", MODEL_PATH)),
        "gaussian": line_ref(MODEL_PATH, line_of(model, "mu,logvar = self.gaussian_parametrizer(features)", MODEL_PATH)),
        "sample": line_ref(MODEL_PATH, line_of(model, "latent_features = self.reparameterize", MODEL_PATH)),
        "classifier": line_ref(MODEL_PATH, line_of(model, "results = self.classifier(latent_features)", MODEL_PATH)),
        "gp_default": line_ref(MSSF_PATH, line_of(mssf, "parser.add_argument('--gp'", MSSF_PATH)),
    }

    report = f"""BioKORF MSSF / KG integration audit
===================================

Scope and conclusion
--------------------
This is a read-only architecture audit. mssf.py, model.py, and models/kg_encoder.py were inspected but not modified. The safest minimal future fusion point is the tensor named `features` immediately after the top-level Attention and immediately before GaussianParametrizer ({refs['final_attention']} -> {refs['gaussian']}). Conceptually this is H_pair with shape [B, 384].

1. MSSF architecture trace and exact dimensions
------------------------------------------------
Notation: B is the current minibatch size. The verified source matrix Datas/drug_side.pkl has shape [757, 994].

Input views:
- The 11 drug views are assembled in read_raw_data and returned at {refs['read_return']}. Each is a 757 x 757 drug-similarity matrix. fold_files horizontally stacks them at {refs['hstack_drug']} into [757, 11*757] = [757, 8327], then selects pair rows with data_train[:,0] at {refs['select_train_drug']}. Model input `drugs` is [B, 8327].
- The 4 side-effect views are returned by the same function. Each is a 994 x 994 side-effect-similarity matrix. They are stacked at {refs['hstack_side']} into [994, 4*994] = [994, 3976], then selected with data_train[:,1] at {refs['select_train_side']}. Model input `sides` is [B, 3976].

EN-con / EncoderConnection ({refs['en_con']}):
- Concatenated input: [B, 8327+3976] = [B, 12303].
- l1: 12303 -> 256; branch Attention: [B,256] -> [B,256]; l2: 256 -> 128.
- Branch output feature1: [B,128]. Decoder reconstruction recCon: 128 -> 256 -> 12303, hence [B,12303].

EN-add / EncoderAddition ({refs['en_add']}):
- drugs.chunk(11,1) gives eleven [B,757] tensors; their elementwise sum is [B,757].
- sides.chunk(4,1) gives four [B,994] tensors; their sum is [B,994].
- Concatenated addition input: [B,1751]. l1: 1751 -> 256; branch Attention: [B,256] -> [B,256]; l2: 256 -> 128.
- Branch output feature2: [B,128]. Decoder reconstruction recAdd: 128 -> 256 -> 1751, hence [B,1751].

CNN-im / CrossProduction ({refs['cnn']}):
- Preprocess maps every drug view 757 -> 128 and every side view 994 -> 128.
- All 11*4=44 outer products create [B,44,128,128].
- Three Conv2d layers use kernel=4, stride=4, no padding: spatial 128 -> 32 -> 8 -> 2, with 32 output channels.
- Flattened CNN output feature3: [B,32*2*2] = [B,128].

Top-level fusion and classifier:
- feature1, feature2, feature3 concatenate at {refs['feature_concat']}: [B,128*3] = [B,384].
- Top-level Attention at {refs['final_attention']}: [B,384] -> [B,384].
- GaussianParametrizer at {refs['gaussian']}: two independent Linear(384,args.gp) layers produce mu and logvar, each [B,args.gp]. The CLI default is args.gp=64 ({refs['gp_default']}).
- Reparameterization at {refs['sample']} returns latent_features [B,args.gp]. In training it samples; in evaluation it returns mu.
- Classifier at {refs['classifier']}: args.gp -> args.gp//2 -> 5. With the default: 64 -> 32 -> 5. Final logits are [B,5].

Loss ({refs['loss']}):
- CrossEntropyLoss over five frequency classes, using label=(frequency.long()-1).
- Plus 0.001 * mean KL divergence.
- Plus 0.0001 * EN-con reconstruction MSE and 0.0001 * EN-add reconstruction MSE.
- Exact total: multi_loss + 0.001*kl_div + 0.0001*rec_loss1 + 0.0001*rec_loss2.

2. Training sample indexing
---------------------------
- Extract_positive_negative_samples allocates interaction_target [757*994,3] = [752458,3] at {refs['sample_alloc']}.
- For every matrix cell, column 0 receives drug row index i ({refs['sample_drug']}), column 1 receives side-effect column index j ({refs['sample_side']}), and column 2 receives frequency DAL[i,j] ({refs['sample_label']}).
- Because DAL is verified [757,994], drug indices are exactly in 0..756 and side-effect indices exactly in 0..993 before filtering. Only nonzero-frequency rows are retained; the local matrix has 37,387 such samples.
- In fold_files, data_train/data_test have conceptual shape [N,3]. Their columns are drug_index, side_index, frequency. These indices select the stacked entity feature matrices ({refs['select_train_drug']}, {refs['select_train_side']}).
- Critical current limitation: TensorDataset at {refs['tensor_dataset']} stores only FloatTensor(drug_features), FloatTensor(side_features), FloatTensor(frequency). It does not retain drug_index or side_index.
- The minibatch at {refs['batch_unpack']} therefore contains batch_drug [B,8327], batch_side [B,3976], and batch_ratings [B]; there are no batch_drug_index or batch_side_index tensors. Consequently Z_drug_KG[drug_index] and Z_side_KG[side_index] cannot be retrieved safely in the current loop without first preserving those integer indices through fold_files and TensorDataset.
- DataLoaders shuffle samples ({refs['dataloader']}), so reconstructing indices from batch position would be incorrect.

3. Candidate H_pair integration point
-------------------------------------
Recommended H_pair:
- Source: model.py Mulmodel.forward, `features = self.attention(features)` at {refs['final_attention']}.
- Shape: [B,384]; feature dimension: 384.
- It is the exact tensor consumed by GaussianParametrizer on the following operation ({refs['gaussian']}). Fusion here places KG evidence before variational parameterization and final classification while preserving all three current MSSF branches.
- Caveat: this point is deterministic with respect to the Bayesian block, but upstream Dropout remains active during model training, and the current Attention mixes samples across the batch. Thus it is not sample-independent in training.

Other reasonable point:
- The concatenated `features` immediately before top-level Attention at {refs['feature_concat']}, shape [B,384]. This avoids adding KG after the final batch-axis Attention, but feature1 and feature2 have already passed through the same batch-axis Attention implementation. It is therefore not fully sample-independent either.
- The separate branch outputs feature1, feature2, feature3, each [B,128], are earlier possible fusion sites, but require branch-specific fusion and are less minimal.

4. Attention-axis audit
-----------------------
Attention.forward receives x [B,D]. At {refs['attention_q']}, each projection is reshaped and transposed:
- Q, K, V: [B,D] -> [B,4,D/4] -> [4,B,D/4].
- Score matrix at {refs['attention_score']}: Q @ K^T -> [4,B,B]. Softmax is over the last B dimension.
- Attention output: [4,B,B] @ [4,B,D/4] -> [4,B,D/4] -> [B,D].

Therefore attention operates ACROSS SAMPLES IN THE MINIBATCH. It does not attend across the 11 drug views, 4 side views, or three branch features within one sample.
- EN-con Attention: D=256, Q/K/V [4,B,64], scores [4,B,B], output [B,256].
- EN-add Attention: D=256, Q/K/V [4,B,64], scores [4,B,B], output [B,256].
- Top-level Attention: D=384, Q/K/V [4,B,96], scores [4,B,B], output [B,384].
This also means a sample's representation depends on its minibatch companions and batch order/composition. The next integration step should preserve behavior initially, but this deserves a separate controlled correction later.

5. Bayesian-block audit
-----------------------
- Bayesian variational inference begins at GaussianParametrizer ({refs['gaussian']}), which receives H_pair/features [B,384] and returns mu/logvar [B,args.gp].
- Stochastic sampling begins at reparameterize ({refs['sample']}); during evaluation it returns mu without sampling.
- KG fusion can safely occur directly before GaussianParametrizer: fused output remains [B,384], so the existing Gaussian and classifier interfaces need not change.
- The BVI block can later be removed without breaking EN-con, EN-add, Preprocess, CrossProduction, or the top-level Attention. To preserve the classifier's current args.gp input, replace GaussianParametrizer + reparameterize with a deterministic Linear(384,args.gp) (or deliberately change the classifier input to 384). The KL term and mu/logvar return contract in mssf.py would also need an explicit coordinated change. Do not simply delete BVI without adjusting those consumers.

6. Recommended future KG fusion (not implemented)
--------------------------------------------------
For each retained pair index, gather:
- H_pair: [B,384]
- Z_drug_KG[drug_index]: [B,128]
- Z_side_KG[side_index]: [B,128]
- drug_kg_available_mask: [B,1]
- side_kg_available_mask: [B,1]

Exact combined KG input is [B,128+128+1+1] = [B,258]. A minimal learned gated residual fusion that preserves the Gaussian input contract is:
1. kg_context = concat(Z_drug_KG, Z_side_KG, drug_mask, side_mask): 258.
2. kg_projected = Linear(258,384): [B,384].
3. gate = sigmoid(Linear(concat(H_pair,kg_context),384)): input 384+258=642, output [B,384].
4. H_fused = LayerNorm(H_pair + gate * kg_projected): [B,384].
5. Feed H_fused to the unchanged GaussianParametrizer(384,args.gp).

This uses dimensions derived from code and the fixed 128-dimensional KG encoder output. It is preferable to passing a raw 642-dimensional concatenation into BVI because it preserves the existing 384-dimensional downstream contract and lets the model suppress KG evidence conditionally. The existing trainable missing-KG embeddings remain inputs, while masks explicitly expose availability.

7. R-GCN training-computation audit
------------------------------------
The minibatch loop is `for i, data in enumerate(train_loader,0)` at {refs['batch_loop']}; default batch size is 128. The KG contains 21,829 nodes, 1,048,598 directed edges, and 40 relations. Recomputing both full R-GCN layers for every MSSF minibatch would repeatedly process the same million-edge graph and is computationally wasteful.

Implementation complexity, lowest to highest:
1. A. Frozen/precomputed KG embeddings: simplest; run encoder once after training/loading it separately and index cached anchor tensors.
2. D. Full R-GCN every minibatch: conceptually simple wiring, but operationally poor and memory/compute intensive.
3. B. Full-graph R-GCN once per epoch: moderate complexity because caching, gradient lifetime, stale embeddings, and optimizer scheduling must be handled deliberately.
4. C. Neighbor-sampled joint training: highest implementation complexity; requires sampled heterogeneous/relational neighborhoods and careful anchor batching.

Expected computational cost, lowest to highest:
1. A. Frozen/precomputed: lowest MSSF training cost, but no joint KG adaptation.
2. C. Neighbor-sampled joint training: generally bounded per minibatch and potentially much cheaper than full-graph recomputation, though sampling overhead and fanout matter.
3. B. Full graph once per epoch: one million-edge two-layer pass per epoch; practical if memory permits, but joint gradient reuse across many MSSF batches needs a sound training schedule.
4. D. Full graph every minibatch: highest cost by far and not recommended.

No training strategy is implemented by this audit.

8. Files/lines for the NEXT step
--------------------------------
mssf.py:
- fold_files around {refs['select_train_drug']} and {refs['select_train_side']}: return the original integer drug and side indices alongside feature rows and labels.
- TensorDataset construction around {refs['tensor_dataset']}: add LongTensor drug_index and side_index fields.
- Training and testing batch unpacking around {refs['batch_unpack']}: receive and pass the two index tensors to the model/fusion layer.
- Loss call around {refs['loss']}: keep unchanged if the model continues returning the current outputs; only revise later if BVI is deliberately removed.

model.py:
- Mulmodel.__init__ around {refs['final_attention']} / the module declarations near {refs['en_con']}: add a dedicated gated KG fusion module with the 258 -> 384 projection and 642 -> 384 gate.
- Mulmodel.forward between {refs['final_attention']} and {refs['gaussian']}: gather already-computed pair KG embeddings by indices, apply gated fusion, and pass [B,384] onward.
- Forward signature at the Mulmodel.forward definition must accept drug_index, side_index, KG anchor embeddings, and availability masks (or a narrowly scoped KG context object).

KG support code:
- models/kg_encoder.py already exposes ordered drug/side anchor extraction and fallback masks. It does not need architectural modification for minimal fusion.
- A new graph-loading/caching integration utility is preferable to putting million-edge CSV loading inside mssf.py or model.py.
- The training orchestrator must choose one of strategies A/B/C; it must never execute a full graph load or R-GCN pass inside each pair-forward call by accident.

9. Final recommendation
-----------------------
First preserve explicit pair indices through the dataset. Then introduce a separate gated fusion module after the top-level MSSF Attention and before GaussianParametrizer, keeping the output at 384 dimensions. Begin with frozen/precomputed KG anchor embeddings for the lowest-risk integration test. Do not alter the current attention or BVI in the same change; audit and ablate those separately so effects remain attributable.
"""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print("BioKORF MSSF integration audit completed")
    print(f"Verified drug-side matrix shape: {drug_side.shape}")
    print("MSSF pair inputs: drug [B,8327], side [B,3976]")
    print("Recommended H_pair: model.py `features` after top-level Attention, [B,384]")
    print("Attention axis: across minibatch samples; score tensors [4,B,B]")
    print("Current minibatches do not retain drug/side indices")
    print("Recommended first strategy: frozen/precomputed KG embeddings + gated 384-d residual fusion")
    print("No source model files were modified; no training was performed")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
