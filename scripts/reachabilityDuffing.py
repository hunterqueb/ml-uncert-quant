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
from torch import nn


from qutils.ml.utils import printModelParmSize, getDevice, Adam_mini
from qutils.tictoc import timer
from qutils.ml.mamba import Mamba, MambaConfig
from qutils.ml.utils import findDecAcc

#import for superweight identification
from qutils.ml.superweight import printoutMaxLayerWeight,getSuperWeight,plotSuperWeight, findMambaSuperActivation,plotSuperActivation

# args parsing for model, horizon, traj_index
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='mamba', help='Model to use')
parser.add_argument('--horizon', type=int, default=1, help='Predict this many steps ahead (target at t+horizon)')
parser.add_argument('--lookback', type=int, default=1, help='Number of past steps fed to the model')
parser.add_argument('--train-timesteps', type=int, default=10, help='Number of time steps from each edge used as training time region')
parser.add_argument('--traj-index', type=int, default=0, help='Trajectory index to plot')
parser.add_argument('--train-ratio', type=float, default=0.8, help='Ratio of trajectories to use for training (rest used for testing)')
parser.add_argument('--batch', type=int, default=256, help='Batch size for training')
parser.add_argument('--batch-test', type=int, default=128, help='Batch size for evaluation')
parser.add_argument('--n-epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for training')
parser.add_argument('--jetson', action='store_true', help='use flag to run on jetson with smaller test size')
parser.add_argument('--dt',type=float,default=0.02,help="numerical value for timestep")
parser.add_argument('--n',type=int,default=20000,help='amount of trajectories used for picking dataset')
parser.add_argument('--pdf', action='store_true', help='Whether to save plots in PDF format instead of PNG')
parser.add_argument("--mu", type=np.array, default=[0.2, 0.0], help="Mean shift to apply to initial conditions")
parser.add_argument("--sigma", type=float, default=0.2, help="Additional uniform noise radius to apply to initial conditions")

# lstm hyperparameters
parser.add_argument('--hidden', type=int, default=64, help='LSTM hidden size')
parser.add_argument('--layers', type=int, default=1, help='Number of LSTM layers')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout (only effective if layers>1, depending on implementation)')
parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay for AdamW')
parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping norm')

args = parser.parse_args()
modelString = args.model
traj_index = args.traj_index


if args.pdf:
    saveType = 'pdf'
else:
    saveType = 'png'

problemDim = 2

device = getDevice()


# hyperparameters
n_epochs = args.n_epochs
lr = args.lr
input_size = problemDim
output_size = problemDim
num_layers = 1
lookback = args.lookback
horizon = args.horizon
train_timesteps = args.train_timesteps

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

# load data
dataFile = f'./data/test/duffing_monte_carlo_trajectories_sigma_{args.sigma}_dt_{args.dt}_n_{args.n}.npz'

system_data = np.load(dataFile,allow_pickle=True)
dt = system_data['dt']
trajs = system_data['trajectories'] # shape (num_trajectories, num_time_steps, problemDim)
t = np.arange(0,trajs.shape[1]*dt,dt)

# reshape numericResult to be (num_time_steps, num_trajectories, problemDim)
trajs_t = np.transpose(trajs, (1, 0, 2))  # (num_time_steps, num_trajectories, problemDim)
num_time_steps = trajs_t.shape[0]
numericResult = trajs_t

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
print(train_in.shape)
print(train_out.shape)
print(test_in.shape)
print(test_out.shape)

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


##############################
# Plotting
##############################


def alpha_shape_segments_and_area(points, radius_quantile=0.65):
    n = points.shape[0]
    if n < 4:
        hull = ConvexHull(points)
        verts = hull.vertices
        cyc = np.column_stack([verts, np.roll(verts, -1)])
        return points[cyc], hull.volume

    try:
        tri = Delaunay(points)
    except QhullError:
        hull = ConvexHull(points)
        verts = hull.vertices
        cyc = np.column_stack([verts, np.roll(verts, -1)])
        return points[cyc], hull.volume

    simplices = tri.simplices
    p = points[simplices]
    a = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
    b = np.linalg.norm(p[:, 2] - p[:, 1], axis=1)
    c = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
    s = 0.5 * (a + b + c)
    area_sq = s * (s - a) * (s - b) * (s - c)
    area_sq = np.maximum(area_sq, 0.0)
    tri_area = np.sqrt(area_sq)

    valid = tri_area > 1e-12
    if not np.any(valid):
        hull = ConvexHull(points)
        verts = hull.vertices
        cyc = np.column_stack([verts, np.roll(verts, -1)])
        return points[cyc], hull.volume

    circum_r = np.full_like(tri_area, np.inf)
    circum_r[valid] = (a[valid] * b[valid] * c[valid]) / (4.0 * tri_area[valid])
    r_thresh = np.quantile(circum_r[valid], radius_quantile)
    keep = valid & (circum_r <= r_thresh)

    if not np.any(keep):
        hull = ConvexHull(points)
        verts = hull.vertices
        cyc = np.column_stack([verts, np.roll(verts, -1)])
        return points[cyc], hull.volume

    kept = simplices[keep]
    edges = np.concatenate(
        [kept[:, [0, 1]], kept[:, [1, 2]], kept[:, [2, 0]]],
        axis=0
    )
    edges = np.sort(edges, axis=1)
    uniq_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = uniq_edges[counts == 1]

    return points[boundary_edges], tri_area[keep].sum()

def alpha_shape_faces_and_volume(points, edge_quantile=0.95):
    n = points.shape[0]
    if n < 5:
        hull = ConvexHull(points)
        return points[hull.simplices], hull.volume

    try:
        tet = Delaunay(points)
    except QhullError:
        hull = ConvexHull(points)
        return points[hull.simplices], hull.volume

    simplices = tet.simplices  # (m, 4)
    p = points[simplices]      # (m, 4, 3)

    e01 = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
    e02 = np.linalg.norm(p[:, 2] - p[:, 0], axis=1)
    e03 = np.linalg.norm(p[:, 3] - p[:, 0], axis=1)
    e12 = np.linalg.norm(p[:, 2] - p[:, 1], axis=1)
    e13 = np.linalg.norm(p[:, 3] - p[:, 1], axis=1)
    e23 = np.linalg.norm(p[:, 3] - p[:, 2], axis=1)
    max_edge = np.maximum.reduce([e01, e02, e03, e12, e13, e23])

    cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    tet_vol = np.abs(np.einsum("ij,ij->i", cross, p[:, 3] - p[:, 0])) / 6.0
    valid = tet_vol > 1e-14
    if not np.any(valid):
        hull = ConvexHull(points)
        return points[hull.simplices], hull.volume

    thresh = np.quantile(max_edge[valid], edge_quantile)
    keep = valid & (max_edge <= thresh)
    if not np.any(keep):
        hull = ConvexHull(points)
        return points[hull.simplices], hull.volume

    kept = simplices[keep]
    faces = np.concatenate(
        [
            kept[:, [0, 1, 2]],
            kept[:, [0, 1, 3]],
            kept[:, [0, 2, 3]],
            kept[:, [1, 2, 3]],
        ],
        axis=0,
    )
    faces_sorted = np.sort(faces, axis=1)
    uniq_faces, counts = np.unique(faces_sorted, axis=0, return_counts=True)
    boundary_faces = uniq_faces[counts == 1]
    return points[boundary_faces], tet_vol[keep].sum()


true_segments, area_true = alpha_shape_segments_and_area(final_true,radius_quantile=0.95)
pred_segments, area_pred = alpha_shape_segments_and_area(final_pred,radius_quantile=0.95)

# calculate reachable-set areas and find ratio of pred area to true area
area_true = float(area_true)
area_pred = float(area_pred)

area_ratio = (area_pred) / area_true if area_true > 0 else float('inf')

print(f"True Alpha-Shape Area: {area_true:.4f}, Pred Alpha-Shape Area: {area_pred:.4f}, Area Ratio (Pred/True): {area_ratio:.4f}")

plt.figure(figsize=(8, 6))
split_index = int(numericResult.shape[1] * args.train_ratio)
train_init = numericResult[0, :split_index, :]  # (num_train_trajs, 2)
test_init = numericResult[0, split_index:, :]   # (num_test_trajs, 2)
plt.scatter(train_init[:, 0], train_init[:, 1], s=10, alpha=0.4, label='Train Initial States', color='blue')
plt.scatter(test_init[:, 0], test_init[:, 1], s=10, alpha=0.1, label='Test Initial States', color='C1')
plt.title(f"Duffing Oscillator Initial Conditions\n Train Size: {train_init.shape[0]}, Test Size: {test_init.shape[0]}")
plt.xlabel('x')
plt.ylabel('xdot')
plt.legend()
plt.savefig("plots/"+modelString+f'_initial_conditions_ratio_{args.train_ratio}_prediction_epoch_{n_epochs}_index_{traj_index}_lr_{lr}_train_timesteps_{train_timesteps}.{saveType}')
plt.close()


fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(true_test_seq[:, 0], true_test_seq[:, 1], 'k-', label='True Trajectory')
tr = train_timesteps + 1
ax.plot(pred_test_seq[:tr, 0], pred_test_seq[:tr, 1], 'b--', label='Pred (train region)')
ax.plot(pred_test_seq[tr-1:, 0], pred_test_seq[tr-1:, 1], 'r--', label='Pred (test region)')

ax.set_title(modelString+' Reachability Prediction: Trajectory Index '+str(traj_index))
ax.set_xlabel('x')
ax.set_ylabel('xdot')
ax.legend(loc='best')
# save plot
plt.grid()
plt.savefig("plots/"+modelString+f'_reachability_ratio_{args.train_ratio}_prediction_epoch_{n_epochs}_index_{traj_index}_lr_{lr}_train_timesteps_{train_timesteps}.{saveType}')
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(final_true[:, 0], final_true[:, 1], s=6, alpha=0.4, label='True Final States')
ax.scatter(final_pred[:, 0], final_pred[:, 1], s=6, alpha=0.4, label='Pred Final States')
for i, seg in enumerate(true_segments):
    ax.plot([seg[0, 0], seg[1, 0]], [seg[0, 1], seg[1, 1]], [0.0, 0.0], c='k', lw=2,
            label='True Alpha Shape' if i == 0 else None)
for i, seg in enumerate(pred_segments):
    ax.plot([seg[0, 0], seg[1, 0]], [seg[0, 1], seg[1, 1]], [0.0, 0.0], c='r', lw=2, ls='--',
            label='Pred Alpha Shape' if i == 0 else None)

ax.set_title(modelString + ' Final-State Alpha Shapes: Area Ratio {:.4f}'.format(area_ratio))
ax.set_xlabel('x')
ax.set_ylabel('xdot')
ax.legend(loc='best')
plt.grid()
plt.savefig("plots/" + modelString + f'_final_state_alpha_shapes_ratio_{args.train_ratio}_epoch_{n_epochs}_lr_{lr}_train_timesteps_{train_timesteps}.{saveType}')
plt.close()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(final_true[:, 0], final_true[:, 1],  s=6, alpha=0.4, label='True Final States')
ax.scatter(final_pred[:, 0], final_pred[:, 1],  s=6, alpha=0.4, label='Pred Final States')

ax.set_title(modelString + ' Final-State Points')
ax.set_xlabel('x')
ax.set_ylabel('xdot')
ax.legend(loc='best')
plt.grid()
plt.savefig("plots/" + modelString + f'_final_state_points_ratio_{args.train_ratio}_epoch_{n_epochs}_lr_{lr}_train_timesteps_{train_timesteps}.{saveType}')
plt.close()
# Plot one random test trajectory: predicted vs true across time
rng = np.random.default_rng(12)
rand_traj_idx = rng.integers(0, test_in.shape[2])  # shape: (num_windows, L, num_trajs, D)
time_axis = np.arange(true_test_seq.shape[0]) * 60.0  # Assuming 60s time step


fig = plt.figure(figsize=(8, 6))
plt.plot(time_axis, true_test_seq[:,0], 'k-', lw=1.5, label='True')
plt.plot(time_axis, pred_test_seq[:,0], 'r--', lw=1.5, label='Predicted')
fig.suptitle(f'Random Test Trajectory #{rand_traj_idx} (Pred vs True)')
plt.tight_layout()
plt.grid()
plt.savefig("plots/" + modelString + f'_random_test_trajectory_{rand_traj_idx}_epoch_{n_epochs}_lr_{lr}_train_timesteps_{train_timesteps}.{saveType}')
plt.close()

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


# Build (num_frames, num_trajs, D) arrays for animation.
# Each array covers the full time range: the initial lookback window states followed by
# per-window predictions, giving a complete reachable-surface animation from t=0.
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

from qutils.ml import getQ2Norm

qNorm = getQ2Norm(true_reach, pred_reach)

# plot Q2 norm over time
plt.figure(figsize=(8, 6))
plt.plot(t,qNorm, 'm-')
plt.xlabel('Time (s)')
plt.ylabel('Q2 Norm')
plt.title(modelString + ' Q2 Norm Over Time')
plt.grid()
# train test boudary line
plt.axvline(x=train_timesteps*float(dt), color='gray', linestyle='--', label='Train/Test Boundary')
plt.legend(loc='best')
plt.savefig("plots/" + modelString + f'_Q2_norm_ratio_{args.train_ratio}_epoch_{n_epochs}_lr_{lr}_train_timesteps_{train_timesteps}.{saveType}')

print(true_reach.shape)
print(pred_reach.shape)

n_frames = min(true_reach.shape[0], pred_reach.shape[0])
true_reach = true_reach[:n_frames]
pred_reach = pred_reach[:n_frames]

x_all = np.concatenate([true_reach[..., 0].ravel(), pred_reach[..., 0].ravel()])
y_all = np.concatenate([true_reach[..., 1].ravel(), pred_reach[..., 1].ravel()])
x_pad = 0.05 * max(1e-9, (x_all.max() - x_all.min()))
y_pad = 0.05 * max(1e-9, (y_all.max() - y_all.min()))

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(x_all.min() - x_pad, x_all.max() + x_pad)
ax.set_ylim(y_all.min() - y_pad, y_all.max() + y_pad)
ax.set_xlabel('x')
ax.set_ylabel('xdot')
ax.set_title('Reachable Set Evolution: ' + modelString)
ax.grid(alpha=0.2, linewidth=0.5)

true_scatter = ax.scatter([], [], s=6, alpha=0.45, c='k', label='True')
pred_scatter = ax.scatter([], [], s=6, alpha=0.45, c='purple', label='Network Output', marker='x')
frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top')
ax.legend(loc='best')
plt.grid()

def _init():
    empty = np.empty((0, 2), dtype=np.float32)
    true_scatter.set_offsets(empty)
    pred_scatter.set_offsets(empty)
    frame_text.set_text('')
    return true_scatter, pred_scatter, frame_text

def _update(frame_idx):
    true_scatter.set_offsets(true_reach[frame_idx])
    pred_scatter.set_offsets(pred_reach[frame_idx])
    if frame_idx < train_timesteps:
        true_scatter.set_facecolor('k')
        pred_scatter.set_facecolor('b')
        frame_text.set_text(f'Train Region\nt = {frame_idx * float(dt):.2f} s')
    else:
        true_scatter.set_facecolor('k')
        pred_scatter.set_facecolor('r')
        frame_text.set_text(f'Test Region\nt = {frame_idx * float(dt):.2f} s')
    return true_scatter, pred_scatter, frame_text

if saveType != "pdf":  # skip animation for PDF output to save time
    anim = FuncAnimation(
        fig,
        _update,
        init_func=_init,
        frames=n_frames,
        interval=70,
        blit=True,
        repeat=False,
    )
    print("Saving reachable set animation...")
    out_base = "plots/" + modelString + f'_reachable_set_evolution_ratio_{args.train_ratio}_epoch_{n_epochs}_lr_{lr}_train_timesteps_{train_timesteps}'
    try:
        anim.save(out_base + '.mp4', writer=FFMpegWriter(fps=20, bitrate=1800))
    except Exception:
        anim.save(out_base + '.gif', writer=PillowWriter(fps=20))
    plt.close(fig)

def compute_kde(pts):
    if pts.shape[0] < 4:
        return np.zeros(XX.shape)
    try:
        kde = gaussian_kde(pts.T)
        return kde(positions).reshape(XX.shape)
    except Exception:
        return np.zeros(XX.shape)

n_grid = 80
x_grid = np.linspace(x_all.min() - x_pad, x_all.max() + x_pad, n_grid)
y_grid = np.linspace(y_all.min() - y_pad, y_all.max() + y_pad, n_grid)
XX, YY = np.meshgrid(x_grid, y_grid)
positions = np.vstack([XX.ravel(), YY.ravel()])


extent = [x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]]
Z0_true = compute_kde(true_reach[0])
Z0_pred = compute_kde(pred_reach[0])
vmax_init = max(Z0_true.max(), Z0_pred.max(), 1e-10)

fig_pdf = plt.figure(figsize=(14, 6))
ax_true_pdf = fig_pdf.add_subplot(121, projection='3d')
ax_pred_pdf = fig_pdf.add_subplot(122, projection='3d')
ax_true_pdf.set_title('True Reachable Set PDF')
ax_pred_pdf.set_title('Predicted Reachable Set PDF')
for _ax in [ax_true_pdf, ax_pred_pdf]:
    _ax.set_xlabel('x')
    _ax.set_ylabel('xdot')
    _ax.set_zlabel('Density')
time_text_pdf = fig_pdf.text(0.5, 0.01, '', ha='center', fontsize=12)
fig_pdf.suptitle('Reachable Set PDF Evolution: ' + modelString)
fig_pdf.tight_layout(rect=[0, 0.04, 1, 1])

def _init_pdf():
    ax_true_pdf.cla()
    ax_pred_pdf.cla()
    time_text_pdf.set_text('')
    return []

def _update_pdf(frame_idx):
    ax_true_pdf.cla()
    ax_pred_pdf.cla()
    Z_true = compute_kde(true_reach[frame_idx])
    Z_pred = compute_kde(pred_reach[frame_idx])
    ax_true_pdf.plot_surface(XX, YY, Z_true, cmap='Blues', alpha=0.85)
    ax_pred_pdf.plot_surface(XX, YY, Z_pred, cmap='Reds', alpha=0.85)
    for _ax in [ax_true_pdf, ax_pred_pdf]:
        _ax.set_xlabel('x')
        _ax.set_ylabel('xdot')
        _ax.set_zlabel('Density')
    region = 'Train' if frame_idx < train_timesteps else 'Test'
    time_text_pdf.set_text(f'{region} Region — t = {frame_idx * float(dt):.2f} s')
    return []

if saveType != "pdf":  # skip animation for PDF output to save time
    print("Saving PDF animation...")
    anim_pdf = FuncAnimation(
        fig_pdf, _update_pdf, init_func=_init_pdf,
        frames=n_frames, interval=70, blit=False, repeat=False,
    )
    out_pdf = ("plots/" + modelString +
            f'_reachable_set_pdf_ratio_{args.train_ratio}_epoch_{n_epochs}'
            f'_lr_{lr}_train_timesteps_{train_timesteps}')
    try:
        anim_pdf.save(out_pdf + '.mp4', writer=FFMpegWriter(fps=20, bitrate=1800))
    except Exception:
        anim_pdf.save(out_pdf + '.gif', writer=PillowWriter(fps=20))
    plt.close(fig_pdf)

# KL divergence animation comparing true vs predicted distributions over time
_eps = 1e-10
kl_values = []
for fi in range(n_frames):
    p = compute_kde(true_reach[fi]).ravel() + _eps
    q = compute_kde(pred_reach[fi]).ravel() + _eps
    p /= p.sum()
    q /= q.sum()
    kl_values.append(float(np.sum(p * np.log(p / q))))

fig_kl, ax_kl = plt.subplots(figsize=(10, 5))
time_axis = [i * float(dt) for i in range(n_frames)]
ax_kl.set_xlim(0, time_axis[-1])
ax_kl.set_ylim(0, max(kl_values) * 1.1 + 1e-10)
ax_kl.set_xlabel('Time (s)')
ax_kl.set_ylabel('KL Divergence (true || pred)')
ax_kl.set_title(f'KL Divergence Over Time: {modelString}')
if train_timesteps < n_frames:
    ax_kl.axvline(x=time_axis[train_timesteps - 1], color='gray', linestyle='--', label='Train/Test boundary')
    ax_kl.legend()
(kl_line,) = ax_kl.plot([], [], color='purple')
kl_time_text = ax_kl.text(0.02, 0.95, '', transform=ax_kl.transAxes, va='top')

def _init_kl():
    kl_line.set_data([], [])
    kl_time_text.set_text('')
    return kl_line, kl_time_text

def _update_kl(frame_idx):
    kl_line.set_data(time_axis[:frame_idx + 1], kl_values[:frame_idx + 1])
    region = 'Train' if frame_idx < train_timesteps else 'Test'
    kl_time_text.set_text(f'{region} — t = {time_axis[frame_idx]:.2f} s  KL = {kl_values[frame_idx]:.4f}')
    return kl_line, kl_time_text

if saveType != "pdf":  # skip animation for PDF output to save time
    print("Saving KL divergence animation...")

    anim_kl = FuncAnimation(
        fig_kl, _update_kl, init_func=_init_kl,
        frames=n_frames, interval=70, blit=True, repeat=False,
    )
    out_kl = ("plots/" + modelString +
            f'_kl_divergence_ratio_{args.train_ratio}_epoch_{n_epochs}'
            f'_lr_{lr}_train_timesteps_{train_timesteps}')
    try:
        anim_kl.save(out_kl + '.mp4', writer=FFMpegWriter(fps=20, bitrate=1800))
    except Exception:
        anim_kl.save(out_kl + '.gif', writer=PillowWriter(fps=20))
    plt.close(fig_kl)

# plot final frame of KL divergence animation as static image
fig_final_kl = plt.figure(figsize=(10, 5))
ax_final_kl = fig_final_kl.add_subplot(111)
ax_final_kl.set_xlim(0, time_axis[-1])
ax_final_kl.set_ylim(0, max(kl_values) * 1.1 + 1e-10)
ax_final_kl.set_xlabel('Time (s)')
ax_final_kl.set_ylabel('KL Divergence (true || pred)')
ax_final_kl.set_title(f'Final KL Divergence: {modelString} — KL = {kl_values[-1]:.4f}')
if train_timesteps < n_frames:
    ax_final_kl.axvline(x=time_axis[train_timesteps - 1], color='gray', linestyle='--', label='Train/Test boundary')
    ax_final_kl.legend()
ax_final_kl.plot(time_axis, kl_values, color='purple')
plt.grid()
plt.savefig("plots/" + modelString + f'_final_kl_divergence_ratio_{args.train_ratio}_epoch_{n_epochs}_lr_{lr}_train_timesteps_{train_timesteps}.{saveType}')
plt.close(fig_final_kl)

# plot final frame of PDF animation as static image
Z_true_final = compute_kde(true_reach[-1])
Z_pred_final = compute_kde(pred_reach[-1])
vmax_final = max(Z_true_final.max(), Z_pred_final.max(), 1e-10)
fig_final_pdf = plt.figure(figsize=(14, 6))
ax_true_3d = fig_final_pdf.add_subplot(121, projection='3d')
ax_pred_3d = fig_final_pdf.add_subplot(122, projection='3d')
ax_true_3d.plot_surface(XX, YY, Z_true_final, cmap='Blues', alpha=0.85)
ax_pred_3d.plot_surface(XX, YY, Z_pred_final, cmap='Reds', alpha=0.85)
ax_true_3d.set_title(f'True Final Reachable Set PDF')
ax_pred_3d.set_title(f'Pred Final Reachable Set PDF')
for _ax in [ax_true_3d, ax_pred_3d]:
    _ax.set_xlabel('x')
    _ax.set_ylabel('xdot')
    _ax.set_zlabel('Density')
fig_final_pdf.suptitle(f'Final Reachable Set PDF: {modelString}')
fig_final_pdf.tight_layout()
plt.savefig("plots/" + modelString + f'_final_reachable_set_pdf_ratio_{args.train_ratio}_epoch_{n_epochs}_lr_{lr}_train_timesteps_{train_timesteps}.{saveType}')
plt.close(fig_final_pdf)