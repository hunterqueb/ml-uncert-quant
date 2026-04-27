import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import torch
import torch.nn.functional as F
import torch.utils.data as data
import argparse
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.qhull import QhullError # import here for p36 compatibility
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from torch import nn


from qutils.ml.utils import printModelParmSize, getDevice, Adam_mini
from qutils.tictoc import timer
from qutils.ml.mamba import Mamba, MambaConfig
from qutils.ml.utils import findDecAcc
from qutils.orbital import dim2NonDim6, nonDim2Dim6

#import for superweight identification
from qutils.ml.superweight import printoutMaxLayerWeight,getSuperWeight,plotSuperWeight, findMambaSuperActivation,plotSuperActivation

# args parsing for model, horizon, traj_index
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='mamba', help='Model to use')
parser.add_argument('--horizon', type=int, default=1, help='Predict this many steps ahead (target at t+horizon)')
parser.add_argument('--lookback', type=int, default=4, help='Number of past steps fed to the model')
parser.add_argument('--train-timesteps', type=int, default=10, help='Number of time steps from each edge used as training time region')
parser.add_argument('--traj-index', type=int, default=124, help='Trajectory index to plot')
parser.add_argument('--train-ratio', type=float, default=0.8, help='Ratio of trajectories to use for training (rest used for testing)')
parser.add_argument('--batch', type=int, default=256, help='Batch size for training')
parser.add_argument('--batch-test', type=int, default=128, help='Batch size for evaluation')
parser.add_argument('--n-epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for training')
parser.add_argument('--jetson', action='store_true', help='use flag to run on jetson with smaller test size')
parser.add_argument('--dim',action="store_true",help="train WITHOUT non dimensional coordinates")
parser.add_argument('--propMin',type=float,default=30,help="propagation time in minutes for picking dataset (used in dataset path and plot titles)")
parser.add_argument('--n',type=int,default=3000,help='amount of trajectories used for picking dataset')
parser.add_argument('--pdf', action='store_true', help='Whether to save plots in PDF format instead of PNG')
parser.add_argument('--orbit', type=str, default='leo', help='Orbit type for picking dataset (used in plot titles)')

parser.add_argument('--hidden', type=int, default=64, help='Hidden size for LSTM')
parser.add_argument('--layers', type=int, default=1, help='Number of layers for LSTM')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout for LSTM')
parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping norm for LSTM')

args = parser.parse_args()
modelString = args.model
traj_index = args.traj_index


if args.pdf:
    saveType = 'pdf'
else:
    saveType = 'png'

problemDim = 6

device = getDevice()


class SimpleLSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=(dropout if num_layers > 1 else 0.0),
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size),
        )

        # Better default init than PyTorch’s raw defaults for regression
        for name, p in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, x):
        # x: (B, T, D)
        if x.ndim != 3:
            raise ValueError(f"Expected x of shape (B, T, D), got {tuple(x.shape)}")

        out, _ = self.lstm(x)       # out: (B, T, H)
        h_last = out[:, -1, :]      # (B, H)
        y = self.head(h_last)       # (B, output_size)
        return y


# hyperparameters
n_epochs = args.n_epochs
lr = args.lr
input_size = problemDim
output_size = problemDim
num_layers = 1
lookback = args.lookback
horizon = args.horizon
train_timesteps = args.train_timesteps


# import gmat dataset
dataset_loc = f"./data/gmat/"+args.orbit+f"/{args.propMin}min-{args.n}"
dataset_file = "/statesArrayNoThrust.npy"

dataset = np.load(dataset_loc+dataset_file)["statesArrayNoThrust"] # (n_traj,min_prop,problemDim)
num_trajs = dataset.shape[0]
num_time_steps = dataset.shape[1]
print(dataset.shape)

# convert to nondim for better ML -- turn off with args
if not args.dim:
    for i in range(num_trajs):
        dataset[i,:,:]=dim2NonDim6(dataset[i,:,:])

trajs_t = np.transpose(dataset, (1, 0, 2))  # (num_time_steps, num_trajectories, problemDim)
numericResult = trajs_t
train_size = 5
test_size = numericResult.shape[1] - train_size

def create_datasets_spatial(data, lookback, horizon, tw=None):
    # Split across dimension 0 (time): first tw steps for train, remainder for test.
    # Trajectory split uses train_ratio across dimension 1.
    seq_length = lookback
    if tw is None:
        tw = train_timesteps
    split_idx = int(data.shape[1] * args.train_ratio)
    time_end = min(num_time_steps, data.shape[0])
    train_time = data[:tw]
    test_time = data[tw:time_end]

    train_data = train_time[:, :split_idx, :]
    if args.jetson: 
        # for jetson testing, use smaller test set to reduce memory requirements for test loss evaluation
        test_data = test_time[:, split_idx:split_idx+1000, :]
    else:
        test_data = test_time[:, split_idx:, :]

    def build_xy(d):
        xs, ys = [], []
        for i in range(len(d) - seq_length - horizon + 1):
            x = d[i:(i + seq_length)]              # (seq_length, num_trajectories, problemDim)
            y = d[i + seq_length + horizon - 1]    # (num_trajectories, problemDim)
            xs.append(x)
            ys.append(y)
        X = np.stack(xs, axis=0)  # (num_windows, seq_length, num_trajectories, problemDim)
        Y = np.stack(ys, axis=0)  # (num_windows, num_trajectories, problemDim)
        return X, Y

    X_train, Y_train = build_xy(train_data)
    X_test, Y_test = build_xy(test_data)
    # Convert to PyTorch tensors (keep on CPU; move batches to GPU in the loop)
    # Shape: (num_windows, seq_length, num_trajectories, problemDim) — no squeeze, preserves (L,B,D) for Mamba
    X_train = torch.tensor(np.array(X_train)).float()
    Y_train = torch.tensor(np.array(Y_train)).float()
    X_test = torch.tensor(np.array(X_test)).float()
    Y_test = torch.tensor(np.array(Y_test)).float()


    return X_train,Y_train,X_test,Y_test

def create_datasets(data_TND, lookback, horizon, train_ratio=0.8, train_timesteps=None, jetson=False):
    """
    data_TND: (T, N, D)
    Returns:
      X_train: (S_train, lookback, D)
      Y_train: (S_train, D)
      X_test : (S_test,  lookback, D)
      Y_test : (S_test,  D)
      norm: dict with mean/std for de/normalization
      meta: dict with window counts and traj counts for extracting per-time slices
    """
    T, N, D = data_TND.shape
    min_required = lookback + horizon
    split_t = train_timesteps if train_timesteps is not None else int(T * train_ratio)
    if split_t < min_required:
        raise ValueError(
            f"train_timesteps must be >= lookback+horizon ({min_required}), got {split_t}"
        )
    if T - split_t < min_required:
        raise ValueError(
            f"Not enough test timesteps after split: T={T}, split_t={split_t}, "
            f"required test timesteps >= {min_required}"
        )

    train = data_TND[:split_t, :, :]   # (Ttr, N, D)
    test  = data_TND[split_t:, :, :]   # (Tte, N, D)

    if jetson:
        test = test[:, :min(test.shape[1], 1000), :]

    def build_xy(block_TND):
        T_, N_, D_ = block_TND.shape
        W = T_ - lookback - horizon + 1
        if W <= 0:
            raise ValueError(f"Not enough timesteps: T={T_}, lookback={lookback}, horizon={horizon}")
        # X: (W, lookback, N_, D_)
        X = np.stack([block_TND[i:i+lookback] for i in range(W)], axis=0)
        # Y at time i+lookback+horizon-1: (W, N_, D_)
        Y = block_TND[lookback + horizon - 1 : lookback + horizon - 1 + W]
        # reshape to samples per trajectory
        X = X.transpose(0, 2, 1, 3).reshape(W * N_, lookback, D_)  # (W*N_, lookback, D_)
        Y = Y.reshape(W * N_, D_)                                  # (W*N_, D_)
        return X, Y, W, N_

    Xtr, Ytr, Wtr, Ntr = build_xy(train)
    Xte, Yte, Wte, Nts = build_xy(test)

    # Normalization from TRAIN only (apply to X and Y)
    mu = Xtr.reshape(-1, D).mean(axis=0)
    sig = Xtr.reshape(-1, D).std(axis=0)
    sig = np.where(sig < 1e-8, 1.0, sig)

    Xtr = (Xtr - mu) / sig
    Ytr = (Ytr - mu) / sig
    Xte = (Xte - mu) / sig
    Yte = (Yte - mu) / sig

    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    Ytr = torch.tensor(Ytr, dtype=torch.float32)
    Xte = torch.tensor(Xte, dtype=torch.float32)
    Yte = torch.tensor(Yte, dtype=torch.float32)

    norm = {"mu": torch.tensor(mu, dtype=torch.float32), "sig": torch.tensor(sig, dtype=torch.float32)}
    meta = {"W_train": Wtr, "N_train": Ntr, "W_test": Wte, "N_test": Nts, "split_t": split_t}
    return Xtr, Ytr, Xte, Yte, norm, meta


if modelString == 'mamba':
    train_in,train_out,test_in,test_out = create_datasets_spatial(numericResult,lookback,horizon,tw=train_timesteps)
else:
    numericalResult = numericResult.transpose(1,0,2) # reshape to (num_trajectories, num_time_steps, problemDim) for LSTM
    train_in, train_out, test_in, test_out, norm, meta = create_datasets(
        numericResult,
        lookback=lookback,
        horizon=horizon,
        train_ratio=args.train_ratio,
        train_timesteps=args.train_timesteps,
        jetson=args.jetson
    )

loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=args.batch)

# initilizing the model, criterion, and optimizer for the data
config = MambaConfig(d_model=problemDim, n_layers=num_layers,d_conv=16)

def returnModel(modelString = 'mamba'):
    if modelString == 'mamba':
        model = Mamba(config).to(device).float()
    elif modelString == 'lstm':
        model = SimpleLSTMRegressor(
            input_size=input_size,
            hidden_size=args.hidden,      # from argparse
            output_size=output_size,
            num_layers=args.layers,
            dropout=args.dropout,
        ).to(device).float()
    printModelParmSize(model)
    return model


model = returnModel(modelString)

optimizer = Adam_mini(model,lr=lr)

criterion = F.smooth_l1_loss
criterion = torch.nn.HuberLoss()


def trainMamba():
    trainTime = timer()
    for epoch in range(n_epochs):

        model.train()
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            # X_batch: (batch, L, num_trajs, D) → reshape to (L, batch*num_trajs, D) for Mamba
            b, L, T, D_sz = X_batch.shape
            X_mamba = X_batch.permute(1, 0, 2, 3).reshape(L, b * T, D_sz)
            y_flat = y_batch.reshape(b * T, D_sz)
            y_pred = model(X_mamba)[-1]  # take last sequence step: (b*T, D)
            loss = criterion(y_pred, y_flat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        with torch.no_grad():
            def eval_batches(x_all, y_all, batch_size=args.batch_test):
                loader_eval = data.DataLoader(
                    data.TensorDataset(x_all, y_all),
                    shuffle=True,
                    batch_size=batch_size,
                )
                preds = []
                targets = []
                total_loss = 0.0
                total_count = 0
                for xb, yb in loader_eval:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    # xb: (batch, L, num_trajs, D) → (L, batch*num_trajs, D) for Mamba
                    b, L, T, D_sz = xb.shape
                    xb_mamba = xb.permute(1, 0, 2, 3).reshape(L, b * T, D_sz)
                    yb_flat = yb.reshape(b * T, D_sz)
                    pred = model(xb_mamba)[-1]  # (b*T, D)
                    batch_loss = criterion(pred, yb_flat).detach()
                    total_loss += batch_loss.item() * (b * T)
                    total_count += b * T
                    preds.append(pred.reshape(b, T, D_sz).cpu())
                    targets.append(yb.cpu())
                pred_all = torch.cat(preds, dim=0)    # (num_windows, num_trajs, D)
                target_all = torch.cat(targets, dim=0)
                rmse = np.sqrt(total_loss / max(total_count, 1))
                return rmse, pred_all, target_all

            train_loss, y_pred_train, y_true_train = eval_batches(train_in, train_out)
            test_loss, y_pred_test, y_true_test = eval_batches(test_in, test_out)

            decAcc, err1 = findDecAcc(y_true_train, y_pred_train, printOut=False)
            decAcc, err2 = findDecAcc(y_true_test, y_pred_test)
            err = np.concatenate((err1,err2),axis=0)

        print("Epoch %d: train loss %.4f, test loss %.4f\n" % (epoch, train_loss, test_loss))

    trainTime.toc()


def trainLSTM():
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )

    use_amp = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


    trainTime = timer()
    best_test = float('inf')
    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0.0
        total_train_count = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.detach().item() * X_batch.shape[0]
            total_train_count += X_batch.shape[0]

        model.eval()
        with torch.no_grad():
            def eval_rmse(x_all, y_all, batch_size):
                loader_eval = data.DataLoader(
                    data.TensorDataset(x_all, y_all),
                    shuffle=False,
                    batch_size=batch_size,
                    pin_memory=True
                )
                se_sum = 0.0
                n_sum = 0
                preds = []
                targets = []
                for xb, yb in loader_eval:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    pred = model(xb)
                    se_sum += torch.sum((pred - yb) ** 2).item()
                    n_sum += yb.numel()
                    preds.append(pred.cpu())
                    targets.append(yb.cpu())
                rmse = np.sqrt(se_sum / max(n_sum, 1))
                return rmse, torch.cat(preds, dim=0), torch.cat(targets, dim=0)

            train_rmse, y_pred_train, y_true_train = eval_rmse(train_in, train_out, args.batch_test)
            test_rmse,  y_pred_test,  y_true_test  = eval_rmse(test_in,  test_out,  args.batch_test)

            # Optional diagnostic metric you already use
            decAcc, err1 = findDecAcc(y_true_train, y_pred_train, printOut=False)
            decAcc, err2 = findDecAcc(y_true_test, y_pred_test)
            err = np.concatenate((err1, err2), axis=0)

        scheduler.step(test_rmse)

        if test_rmse < best_test:
            best_test = test_rmse

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d}: train RMSE {train_rmse:.6f}, test RMSE {test_rmse:.6f}, lr {lr_now:.2e}")

    trainTime.toc()

if modelString.startswith('mamba'):
    trainMamba()
elif modelString.startswith('lstm'):
    trainLSTM()


def mambaEval():
    def build_full_seq(x_all, y_all, traj_idx):
        x_np = x_all.numpy()
        y_np = y_all.numpy()
        if x_np.ndim == 4:
            init = x_np[0, :, traj_idx, :]
        else:
            init = x_np[0, traj_idx, :][np.newaxis, :]
        y_seq = y_np[:, traj_idx, :]
        print("init shape:", init.shape)
        print("y_seq shape:", y_seq.shape)
        return np.concatenate([init, y_seq], axis=0)

    with torch.no_grad():
        test_loader = data.DataLoader(data.TensorDataset(test_in, test_out), shuffle=False, batch_size=args.batch_test)

        xb, yb = next(iter(test_loader))
        # xb: (batch, L, num_trajs, D) → (L, batch*num_trajs, D) for Mamba
        _b, _L, _T, _D = xb.shape
        xb_mamba = xb.permute(1, 0, 2, 3).reshape(_L, _b * _T, _D).to(device)
        pred = model(xb_mamba)[-1].cpu().numpy()   # (batch*num_trajs, D)
        yb = yb.cpu().numpy()
        xb = xb.cpu().numpy()
        traj_idx = traj_index
        def predict_last_step(x_all, batch_size=args.batch_test, slice_traj_idx=None):
            loader_eval = data.DataLoader(
                data.TensorDataset(x_all),
                shuffle=False,
                batch_size=batch_size,
            )
            preds = []
            for (xb_eval,) in loader_eval:
                xb_eval = xb_eval.to(device)
                # xb_eval: (batch, L, num_trajs, D) → (L, batch*num_trajs, D) for Mamba
                b, L, T, D_sz = xb_eval.shape
                xb_mamba = xb_eval.permute(1, 0, 2, 3).reshape(L, b * T, D_sz)
                pred = model(xb_mamba)[-1].cpu()  # (b*T, D)
                pred = pred.reshape(b, T, D_sz)   # (batch, num_trajs, D)
                if slice_traj_idx is not None:
                    pred = pred[:, slice_traj_idx, :]  # (batch, D)
                preds.append(pred)
            return torch.cat(preds, dim=0)  # (num_windows, num_trajs, D) or (num_windows, D)

        test_pred_full = predict_last_step(test_in)

        traj_split_idx = int(numericResult.shape[1] * args.train_ratio)
        train_traj_prefix = numericResult[:train_timesteps, traj_split_idx + traj_idx, :]  # (train_timesteps, D)
        true_test_seq = np.concatenate(
            [train_traj_prefix, build_full_seq(test_in, test_out, traj_idx)], axis=0
        )  # (800, D)
        pred_test_seq = np.concatenate(
            [train_traj_prefix, build_full_seq(test_in, test_pred_full, traj_idx)], axis=0
        )  # (800, D)

        final_true = test_out[-1].numpy()
        final_pred = test_pred_full[-1].numpy()

        return true_test_seq, pred_test_seq, final_true, final_pred, test_pred_full
    
def lstmEval():
    with torch.no_grad():
        test_loader = data.DataLoader(data.TensorDataset(test_in, test_out), shuffle=False, batch_size=args.batch_test)
        xb, yb = next(iter(test_loader))
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb).cpu().numpy()
        yb = yb.cpu().numpy()
        def predict_last_step(x_all, batch_size=args.batch_test, slice_traj_idx=None):
            loader_eval = data.DataLoader(
                data.TensorDataset(x_all),
                shuffle=False,
                batch_size=batch_size,
            )
            preds = []
            for (xb_eval,) in loader_eval:
                xb_eval = xb_eval.to(device)
                pred = model(xb_eval).cpu()  # (batch, D)
                if slice_traj_idx is not None:
                    pred = pred[:, slice_traj_idx]  # (batch,)
                preds.append(pred)
            return torch.cat(preds, dim=0)  # (num_windows, D)

        test_pred_full = predict_last_step(test_in)

        # De-normalize helper
        mu = norm["mu"]
        sig = norm["sig"]

        def denorm(x):
            # x: torch or np
            if isinstance(x, np.ndarray):
                return x * sig.numpy() + mu.numpy()
            return x * sig + mu

        # Extract last window slice (time = final available) across ALL test trajectories
        Wte = meta["W_test"]
        Nts = meta["N_test"]
        start = (Wte - 1) * Nts
        end = Wte * Nts

        model.eval()
        with torch.no_grad():
            xb_last = test_in[start:end].to(device)
            pred_last = model(xb_last).cpu()
            true_last = test_out[start:end].cpu()

        def build_full_seq(x_all, y_all, traj_idx):
            n_test = meta["N_test"]
            if traj_idx < 0 or traj_idx >= n_test:
                raise IndexError(f"traj_idx out of range: {traj_idx}, expected [0, {n_test-1}]")
            # Flattened layout is (window0 traj0..trajN-1, window1 traj0..trajN-1, ...)
            x_init = x_all[traj_idx, :, :].cpu()        # (lookback, D)
            y_seq = y_all[traj_idx::n_test, :].cpu()    # (num_windows, D)
            full_seq = torch.cat([x_init, y_seq], dim=0)
            return denorm(full_seq).numpy()

        train_traj_prefix = numericResult[:train_timesteps, traj_index, :]  # (train_timesteps, D)
        true_test_seq = np.concatenate(
            [train_traj_prefix, build_full_seq(test_in, test_out, traj_index)], axis=0
        )  # (800, D)
        pred_test_seq = np.concatenate(
            [train_traj_prefix, build_full_seq(test_in, test_pred_full, traj_index)], axis=0
        )  # (800, D)

        final_true = denorm(true_last).numpy()
        final_pred = denorm(pred_last).numpy()

        test_pred_full = denorm(predict_last_step(test_in))

        return true_test_seq, pred_test_seq, final_true, final_pred, test_pred_full

# generate predictions
model.eval()
if modelString.startswith('mamba'):
    true_test_seq, pred_test_seq, final_true, final_pred, test_pred_full = mambaEval()
elif modelString.startswith('lstm'):
    true_test_seq, pred_test_seq, final_true, final_pred, test_pred_full = lstmEval()



if modelString.startswith('mamba'):
    test_loader = data.DataLoader(data.TensorDataset(test_in, test_out), shuffle=False, batch_size=args.batch_test)
    xb, yb = next(iter(test_loader))
    # xb: (batch, L, num_trajs, D) — extract one trajectory and reshape to (L, batch, D)
    b, L, T, D_sz = xb.shape
    xb_one_traj = xb[:, :, traj_index:traj_index+1, :]  # (batch, L, 1, D)
    xb_one_traj = xb_one_traj.permute(1, 0, 2, 3).reshape(L, b, D_sz)  # (L, batch, D)
    magnitude, index = findMambaSuperActivation(model, xb_one_traj.to(device))

    normedMagsMRP = np.zeros((len(magnitude),))
    for i in range(len(magnitude)):
        normedMagsMRP[i] = magnitude[i].norm().detach().cpu()

    printoutMaxLayerWeight(model)
    getSuperWeight(model)
    plotSuperWeight(model)
    plotSuperActivation(magnitude, index,printOutValues=True)
    plt.title("Mamba Reachability Super Activations")
    plt.savefig("plots/" + modelString + f'_super_activations_ratio_{args.train_ratio}_epoch_{n_epochs}_index_{traj_index}_lr_{lr}_train_timesteps_{train_timesteps}.{saveType}')


# construct full reachability sequences for true and predicted, by prepending the initial lookback states to the windowed predictions, for both train and test trajectories. This is needed to compute metrics like decAcc that depend on the full sequence of states, and also for plotting the reachability tube over time for a single trajectory.
if modelString.startswith('mamba'):
    init_reach = test_in.numpy()[0]                          # (lookback, num_trajs, D)
    true_reach_test = np.concatenate(
        [init_reach, test_out.detach().cpu().numpy()], axis=0
    )                                                        # (lookback+num_windows, num_test_trajs, D)
    pred_reach_test = np.concatenate(
        [init_reach, test_pred_full.detach().cpu().numpy()], axis=0
    )                                                        # (lookback+num_windows, num_test_trajs, D)
    traj_split_idx = int(numericResult.shape[1] * args.train_ratio)
    n_test_trajs = true_reach_test.shape[1]  # respects jetson 1000-traj limit
    train_prefix = numericResult[:train_timesteps, traj_split_idx:traj_split_idx + n_test_trajs, :]
    true_reach = np.concatenate([train_prefix, true_reach_test], axis=0)
    # Run model on test-trajectory training-time windows to build predicted train prefix
    test_trajs_train_time = numericResult[:train_timesteps, traj_split_idx:traj_split_idx + n_test_trajs, :]
    W_tr_pred = train_timesteps - lookback - horizon + 1
    if W_tr_pred > 0:
        xs_tr = [test_trajs_train_time[i:i + lookback] for i in range(W_tr_pred)]
        X_tr_pred = torch.tensor(np.stack(xs_tr, axis=0)).float()  # (W_tr_pred, lookback, n_test_trajs, D)
        with torch.no_grad():
            model.eval()
            tr_preds = []
            for (xb_eval,) in data.DataLoader(data.TensorDataset(X_tr_pred), shuffle=False, batch_size=args.batch_test):
                xb_eval = xb_eval.to(device)
                b, L, T, D_sz = xb_eval.shape
                xb_mamba = xb_eval.permute(1, 0, 2, 3).reshape(L, b * T, D_sz)
                tr_preds.append(model(xb_mamba)[-1].cpu().reshape(b, T, D_sz))
            train_pred_wins = torch.cat(tr_preds, dim=0).numpy()  # (W_tr_pred, n_test_trajs, D)
        pred_train_prefix = np.concatenate([test_trajs_train_time[:lookback], train_pred_wins], axis=0)
    else:
        pred_train_prefix = train_prefix
    pred_reach = np.concatenate([pred_train_prefix, pred_reach_test], axis=0)
elif modelString.startswith('lstm'):
    W_te = meta["W_test"]
    N_ts = meta["N_test"]
    mu_np = norm["mu"].numpy()
    sig_np = norm["sig"].numpy()
    # Initial window: first N_ts rows cover all trajs at window 0
    init_np = test_in[:N_ts].numpy()                        # (N_ts, lookback, D) normalized
    init_reach = init_np.transpose(1, 0, 2) * sig_np + mu_np  # (lookback, N_ts, D) denormalized
    true_reach_wins = (test_out.detach().cpu().numpy() * sig_np + mu_np).reshape(W_te, N_ts, -1)
    pred_reach_wins = test_pred_full.detach().cpu().numpy().reshape(W_te, N_ts, -1)
    true_reach_test = np.concatenate([init_reach, true_reach_wins], axis=0)
    pred_reach_test = np.concatenate([init_reach, pred_reach_wins], axis=0)
    split_t = meta["split_t"]
    train_prefix = numericResult[:split_t, :N_ts, :]
    true_reach = np.concatenate([train_prefix, true_reach_test], axis=0)
    # Run model on test-trajectory training-time windows to build predicted train prefix
    test_trajs_train = numericResult[:split_t, :N_ts, :]  # (split_t, N_ts, D)
    W_tr_pred = split_t - lookback - horizon + 1
    if W_tr_pred > 0:
        X_tr = np.stack([test_trajs_train[i:i + lookback] for i in range(W_tr_pred)], axis=0)
        X_tr_flat = X_tr.transpose(0, 2, 1, 3).reshape(W_tr_pred * N_ts, lookback, -1)
        X_tr_flat = (X_tr_flat - mu_np) / sig_np
        X_tr_t = torch.tensor(X_tr_flat, dtype=torch.float32)
        with torch.no_grad():
            model.eval()
            tr_preds = []
            for (xb_eval,) in data.DataLoader(data.TensorDataset(X_tr_t), shuffle=False, batch_size=args.batch_test):
                xb_eval = xb_eval.to(device)
                tr_preds.append(model(xb_eval).cpu())
            train_pred_flat = torch.cat(tr_preds, dim=0).numpy() * sig_np + mu_np  # denormalized
        train_pred_wins_tr = train_pred_flat.reshape(W_tr_pred, N_ts, -1)
        pred_train_prefix = np.concatenate([test_trajs_train[:lookback], train_pred_wins_tr], axis=0)
    else:
        pred_train_prefix = train_prefix
    pred_reach = np.concatenate([pred_train_prefix, pred_reach_test], axis=0)

# plots

# plot projections of true and predicted reachability tubes for the selected trajectory index
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
traj_idx = traj_index
ax.plot(true_reach[:, traj_idx, 0], true_reach[:, traj_idx, 1], true_reach[:, traj_idx, 2], label='True Reachability', color='blue')
ax.plot(pred_reach[:, traj_idx, 0], pred_reach[:, traj_idx, 1], pred_reach[:, traj_idx, 2], label='Predicted Reachability', color='orange')
ax.set_title(f"{modelString.upper()} Reachability Prediction for Trajectory {traj_idx}\nOrbit: {args.orbit.upper()}, Propagation: {args.propMin}min, Train Ratio: {args.train_ratio}, Epochs: {n_epochs}")
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.legend()
plt.savefig("plots/" + modelString + f'_reachability_traj_{traj_idx}_orbit_{args.orbit}_prop{args.propMin}min_trainRatio_{args.train_ratio}_epoch_{n_epochs}.{saveType}')

# plot each state component over time for the selected trajectory index
time_steps = np.arange(true_reach.shape[0])
state_labels = ['X (km)', 'Y (km)', 'Z (km)', 'Vx (km/s)', 'Vy (km/s)', 'Vz (km/s)']
fig, axs = plt.subplots(3, 2, figsize=(15, 10))
for i in range(6):
    ax = axs[i // 2, i % 2]
    ax.plot(time_steps, true_reach[:, traj_idx, i], label='True', color='blue')
    ax.plot(time_steps, pred_reach[:, traj_idx, i], label='Predicted', color='orange')
    ax.set_title(f"{state_labels[i]} over Time for Trajectory {traj_idx}")
    ax.set_xlabel('Time Step')
    ax.set_ylabel(state_labels[i])
    ax.legend()
plt.tight_layout()
plt.savefig("plots/" + modelString + f'_state_components_traj_{traj_idx}_orbit_{args.orbit}_prop{args.propMin}min_trainRatio_{args.train_ratio}_epoch_{n_epochs}.{saveType}')
