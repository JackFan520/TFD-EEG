import sys
import os
import torch
import torch.nn as nn
from utils.utils import *
from config.config import *
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# Check CUDA availability and set device
CUDA = torch.cuda.is_available()
_, os.environ['CUDA_VISIBLE_DEVICES'] = set_config()

def train_one_epoch(data_loader, net, loss_fn, optimizer):
    net.train()
    tl = Averager()
    pred_train = []
    act_train = []
    for i, (x_batch, y_batch) in enumerate(data_loader):
        if CUDA:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

        out = net(x_batch)
        loss = loss_fn(out, y_batch)
        _, pred = torch.max(out, 1)
        pred_train.extend(pred.data.tolist())
        act_train.extend(y_batch.data.tolist())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tl.add(loss.item())
    return tl.item(), pred_train, act_train


def predict(data_loader, net, loss_fn):
    net.eval()
    pred_val = []
    act_val = []
    vl = Averager()
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(data_loader):
            if CUDA:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

            out = net(x_batch)
            loss = loss_fn(out, y_batch)
            _, pred = torch.max(out, 1)
            vl.add(loss.item())
            pred_val.extend(pred.data.tolist())
            act_val.extend(y_batch.data.tolist())
    return vl.item(), pred_val, act_val


def set_up(args):
    """
    Set up GPU, random seed, and save path.
    """
    set_gpu(args.gpu)
    ensure_path(args.save_path)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True


def train(args, data_train, label_train, data_val, label_val, subject, fold):
    """
    Train model for one stage with early stopping.
    Records train and validation metrics to files.
    """
    seed_all(args.random_seed)
    save_name = f'_sub{subject}_fold{fold}'
    set_up(args)

    train_loader = get_dataloader(data_train, label_train, args.batch_size)
    val_loader = get_dataloader(data_val, label_val, args.batch_size)

    model = get_model(args)
    if CUDA:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = LabelSmoothing(args.LS_rate) if args.LS else nn.CrossEntropyLoss()

    trlog = {
        'args': vars(args),
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'max_acc': 0.0,
        'F1': 0.0
    }

    timer = Timer()
    patient = args.patient
    counter = 0

    # Paths for logging results
    train_res_file = os.path.join(args.save_path, 'train_result.txt')
    val_res_file = os.path.join(args.save_path, 'validation_result.txt')

    for epoch in range(1, args.max_epoch + 1):
        # Training step
        loss_train, pred_train, act_train = train_one_epoch(
            data_loader=train_loader, net=model, loss_fn=loss_fn, optimizer=optimizer)
        acc_train, f1_train, _ = get_metrics(y_pred=pred_train, y_true=act_train)
        print(f'epoch {epoch}, for the train set, loss={loss_train:.4f} acc={acc_train:.4f} f1={f1_train:.4f}')
        with open(train_res_file, 'a', encoding='utf-8') as f:
            f.write(f"SUB:{subject} FOLD:{fold}的训练集周期epoch {epoch}, for the train set, "
                    f"loss={loss_train:.4f} acc={acc_train:.4f} f1={f1_train:.4f}\n")

        # Validation step
        loss_val, pred_val, act_val = predict(
            data_loader=val_loader, net=model, loss_fn=loss_fn)
        acc_val, f1_val, _ = get_metrics(y_pred=pred_val, y_true=act_val)
        print(f'epoch {epoch}, for the validation set, loss={loss_val:.4f} acc={acc_val:.4f} f1={f1_val:.4f}')
        with open(val_res_file, 'a', encoding='utf-8') as f:
            f.write(f"SUB:{subject} FOLD:{fold}的验证集周期epoch {epoch}, for the validation set, "
                    f"loss={loss_val:.4f} acc={acc_val:.4f} f1={f1_val:.4f}\n")

        # Early stopping logic
        if acc_val >= trlog['max_acc']:
            trlog['max_acc'] = acc_val
            trlog['F1'] = f1_val
            # Save candidate model
            candidate_path = os.path.join(args.save_path, 'candidate.pth')
            torch.save(model.state_dict(), candidate_path)
            counter = 0
        else:
            counter += 1
            if counter >= patient:
                print('early stopping')
                break

        # Update training log
        trlog['train_loss'].append(loss_train)
        trlog['train_acc'].append(acc_train)
        trlog['val_loss'].append(loss_val)
        trlog['val_acc'].append(acc_val)
        print(f'ETA:{timer.measure()}/{timer.measure(epoch/args.max_epoch)} SUB:{subject} FOLD:{fold}')

    # Save training log object
    save_name = 'trlog' + save_name
    experiment_setting = f'T_{args.T}_pool_{args.pool}'
    log_dir = os.path.join(args.save_path, experiment_setting, 'log_train')
    ensure_path(log_dir)
    torch.save(trlog, os.path.join(log_dir, save_name))

    return trlog['max_acc'], trlog['F1']

'''原test函数'''
# def test(args, data, label, reproduce, subject, fold):
#     """
#     Test the model and record results.
#     """
#     set_up(args)
#     seed_all(args.random_seed)
#
#     test_loader = get_dataloader(data, label, args.batch_size)
#     model = get_model(args)
#     if CUDA:
#         model = model.cuda()
#     loss_fn = nn.CrossEntropyLoss()
#
#     # Load the correct model weights
#     if reproduce:
#         model_name = f'sub{subject}_fold{fold}.pth'
#         data_type = f'model_{args.data_format}_{args.label_type}'
#         experiment_setting = f'T_{args.T}_pool_{args.pool}'
#         load_path = os.path.join(args.save_path, experiment_setting, data_type, model_name)
#         model.load_state_dict(torch.load(load_path))
#     else:
#         model.load_state_dict(torch.load(args.load_path_final))
#
#     # Run prediction on the test set
#     loss_test, pred_test, act_test = predict(
#         data_loader=test_loader, net=model, loss_fn=loss_fn)
#     acc, f1, cm = get_metrics(y_pred=pred_test, y_true=act_test)
#     print(f'>>> Test:  loss={loss_test:.4f} acc={acc:.4f} f1={f1:.4f}')
#
#     # Write to test_result.txt
#     test_res_file = os.path.join(args.save_path, 'test_result.txt')
#     with open(test_res_file, 'a', encoding='utf-8') as f:
#         f.write(
#             f"SUB:{subject} FOLD:{fold}的测试集Test:  "
#             f"loss={loss_test:.4f} acc={acc:.4f} f1={f1:.4f}\n"
#         )
#
#     # —— 新增：提取最后一层全连接权重 → 展平 → 记录到 param_log.txt —— #
#     param_file = os.path.join(args.save_path, 'param_log.txt')
#     # model.fc 是 Sequential([Dropout, Linear])
#     linear = model.fc[1]                    # 取出 nn.Linear 层
#     weight_vec = linear.weight.data.cpu().numpy().flatten()
#     # 格式化成逗号分隔，保留 6 位小数
#     vec_str = ",".join(f"{v:.6f}" for v in weight_vec)
#     with open(param_file, 'a', encoding='utf-8') as pf:
#         pf.write(f"SUB:{subject} FOLD:{fold} PARAMS: {vec_str}\n")
#
#     return acc, pred_test, act_test

'''修改后的test函数'''
def test(args, data, label, reproduce, subject, fold):
    """
    Test the model and record results.

    保持原有流程不变，新增：
      - 将 model.fc 最后一层权重写入 param_log.txt（原有）
      - 生成 orig_loss_param_log.txt，但这次把单一 loss 扩展为“波形向量”：
          * 若提供 music_rms（npy/npz）或 args.music_file，则使用音乐 RMS 作为基底并乘以 loss + 少量噪声
          * 否则回退为 repeat_sin/repeat
      - 可选地（当 args.music_file 可用且 librosa 已安装）注册 hook 收集 activation，
        计算 activation 的包络并与音乐比较，保存对比分图与 MSE/Pearson。
    返回： acc, pred_test, act_test
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr

    # librosa 在可选功能中使用（若未安装则跳过 activation 对齐）
    try:
        import librosa
    except Exception:
        librosa = None

    # 原有准备
    set_up(args)
    seed_all(args.random_seed)

    test_loader = get_dataloader(data, label, args.batch_size)
    model = get_model(args)
    if CUDA:
        model = model.cuda()
    loss_fn = nn.CrossEntropyLoss()

    # Load model weights（与你当前逻辑一样）
    if reproduce:
        model_name = f'sub{subject}_fold{fold}.pth'
        data_type = f'model_{args.data_format}_{args.label_type}'
        experiment_setting = f'T_{args.T}_pool_{args.pool}'
        load_path = os.path.join(args.save_path, experiment_setting, data_type, model_name)
        model.load_state_dict(torch.load(load_path))
    else:
        model.load_state_dict(torch.load(args.load_path_final))

    # 可选 hook 收集 activation（仅当 args.music_file 存在且 librosa 可用）
    hook_enabled = (hasattr(args, 'music_file') and args.music_file is not None and librosa is not None)
    collected = []
    handle = None

    def hook_fn(module, input, output):
        try:
            arr = output.detach().cpu().numpy()
        except Exception:
            return
        collected.append(arr)

    if hook_enabled:
        target_module = None
        for name in ['sliding_window_processor', 'feature_integrator', 'Tception1', 'Tception2', 'Tception3']:
            if hasattr(model, name):
                target_module = getattr(model, name)
                break
        if target_module is not None:
            try:
                handle = target_module.register_forward_hook(hook_fn)
            except Exception:
                handle = None
                hook_enabled = False
                print("WARN: 注册 hook 失败，跳过 activation 收集。")
        else:
            hook_enabled = False
            print("WARN: 未找到 time-aware module 来注册 hook，跳过 activation 收集。")

    # ==== 原始预测 ====
    loss_test, pred_test, act_test = predict(data_loader=test_loader, net=model, loss_fn=loss_fn)
    acc, f1, cm = get_metrics(y_pred=pred_test, y_true=act_test)
    print(f'>>> Test:  loss={loss_test:.4f} acc={acc:.4f} f1={f1:.4f}')

    # 写 test 结果（不变）
    test_res_file = os.path.join(args.save_path, 'test_result.txt')
    with open(test_res_file, 'a', encoding='utf-8') as f:
        f.write(
            f"SUB:{subject} FOLD:{fold}的测试集Test:  "
            f"loss={loss_test:.4f} acc={acc:.4f} f1={f1:.4f}\n"
        )

    # 原有：保存最后一层全连接权重到 param_log.txt
    try:
        param_file = os.path.join(args.save_path, 'param_log.txt')
        linear = model.fc[1]  # 取出 nn.Linear 层
        weight_vec = linear.weight.data.cpu().numpy().flatten()
        vec_str = ",".join(f"{v:.6f}" for v in weight_vec)
        with open(param_file, 'a', encoding='utf-8') as pf:
            pf.write(f"SUB:{subject} FOLD:{fold} PARAMS: {vec_str}\n")
    except Exception as e:
        print("WARN: 写入 param_log 失败：", e)

    # ---------------- 关键修改：把 loss 扩展为“波形向量”并写入 orig_loss_param_log.txt ----------------
    def _load_music_rms_if_possible(path):
        # 支持 npy / npz
        if path is None:
            return None
        try:
            if path.endswith('.npy'):
                arr = np.load(path)
                return np.array(arr, dtype=float)
            elif path.endswith('.npz'):
                d = np.load(path)
                # 优先取 'rms_z' 或 'rms'，否则取第一个数组
                for k in ('rms_z', 'rms'):
                    if k in d:
                        return np.array(d[k], dtype=float)
                key = list(d.keys())[0]
                return np.array(d[key], dtype=float)
            else:
                # 不是二进制 np 文件，尝试当作 wav/mp3（需 librosa）
                if librosa is not None and os.path.exists(path):
                    y, sr = librosa.load(path, sr=None)
                    # 计算 RMS
                    frame_length = getattr(args, 'rms_frame_length', 1024)
                    hop_length = getattr(args, 'rms_hop_length', 512)
                    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
                    return np.array(rms, dtype=float)
        except Exception as e:
            print("WARN: load music rms failed:", e)
        return None

    def _write_orig_loss_param_line_wave(save_dir, subject, fold, loss_val,
                                         music_rms_path=None,
                                         mode='music_scaled',
                                         default_M=1024,
                                         noise_amp=0.05):
        """
        生成 orig_loss_param_log.txt 的一行（格式与 param_log.txt 一致），但把 scalar loss 扩展为波形：
          - mode='music_scaled'（优先）：若给定 music_rms_path 且能读取，则以 music_rms_z * loss + noise
          - mode='music_envelope_noise'：用 music envelope 乘以 loss，同时在频谱上加入小随机噪声（不实现复杂谱着色，仅在时域加噪）
          - fallback: 'repeat_sin' 或 'repeat'
        noise_amp: 相对于 |loss| 的噪声比例（例如 0.05 表示噪声标准差为 0.05*|loss|）
        """
        M = default_M
        music_rms = None
        if music_rms_path is not None:
            music_rms = _load_music_rms_if_possible(music_rms_path)
            if music_rms is not None:
                M = len(music_rms)

        base = float(loss_val)

        if mode == 'music_scaled' and music_rms is not None:
            # 使用音乐的 RMS（或 envelope）作为基底
            # 标准化为 z-score（零均值单位方差），再乘以 base
            mus = np.array(music_rms, dtype=float)
            mus_z = (mus - mus.mean()) / (mus.std() + 1e-9)
            # 生成噪声，幅值与 base 相关，避免全零方差
            rng = np.random.RandomState( int((subject+1)*(fold+1)) % 2**32 )
            noise = rng.normal(loc=0.0, scale=(abs(base) * noise_amp + 1e-9), size=mus_z.shape)
            vec = mus_z * base + noise
        elif mode == 'music_envelope_noise' and music_rms is not None:
            # 另一种策略：先把音乐 envelope 平滑（savgol），然后用噪声叠加
            mus = np.array(music_rms, dtype=float)
            mus_z = (mus - mus.mean()) / (mus.std() + 1e-9)
            rng = np.random.RandomState( int((subject+2)*(fold+3)) % 2**32 )
            noise = rng.normal(loc=0.0, scale=(abs(base) * noise_amp + 1e-9), size=mus_z.shape)
            vec = mus_z * base + noise
        elif mode == 'repeat':
            vec = np.ones(M, dtype=float) * base
        elif mode == 'resample_to_music' and music_rms is not None:
            # 单值退化为 repeat（没有历史 loss 序列可插值）
            vec = np.ones(M, dtype=float) * base
        else:
            # fallback 为 repeat_sin（老行为）
            t = np.linspace(0, 1, M)
            amp = max(1e-6, abs(base))
            vec = base * (1.0 + 0.02 * np.sin(2.0 * np.pi * 3.0 * t))  # 保留微小正弦成分

        # 保证数值不会完全恒定（零方差），以便后续 zscore / plotting 不出问题
        if np.allclose(np.std(vec), 0.0, atol=1e-12):
            vec = vec + np.random.RandomState(0).normal(loc=0.0, scale=1e-9, size=vec.shape)

        out_path = os.path.join(save_dir, 'orig_loss_param_log.txt')
        vals = ",".join(f"{v:.6f}" for v in vec)
        line = f"SUB:{subject} FOLD:{fold} PARAMS: {vals}\n"
        with open(out_path, 'a', encoding='utf-8') as f:
            f.write(line)
        return out_path

    # 使用 args 中配置
    orig_mode = getattr(args, 'orig_loss_mode', 'music_scaled')  # 改为默认优先使用 music_scaled
    music_rms_path = getattr(args, 'music_rms', None)  # 可传入 .npy/.npz 路径，或音乐文件路径
    noise_amp = getattr(args, 'orig_loss_noise_amp', 0.05)
    default_M = getattr(args, 'orig_loss_M', 1024)
    try:
        _write_orig_loss_param_line_wave(save_dir=args.save_path, subject=subject, fold=fold,
                                         loss_val=loss_test, music_rms_path=music_rms_path,
                                         mode=orig_mode, default_M=default_M,
                                         noise_amp=noise_amp)
    except Exception as e:
        print("WARN: 生成 orig_loss_param_log 失败：", e)

    # ========== 若启用了 hook，处理 collected activations，与音乐做对齐并保存图 ==========
    if hook_enabled:
        if handle is not None:
            try:
                handle.remove()
            except Exception:
                pass

        # 整理 collected activations -> 每个 sample 一个 ndarray
        sample_acts = []
        for batch_out in collected:
            if isinstance(batch_out, list):
                for item in batch_out:
                    sample_acts.append(np.array(item))
            else:
                arr = np.array(batch_out)
                if arr.ndim >= 3:
                    for s in range(arr.shape[0]):
                        sample_acts.append(arr[s])
                else:
                    sample_acts.append(arr)

        n_pred = len(pred_test) if hasattr(pred_test, '__len__') else None
        if n_pred is not None and len(sample_acts) < n_pred:
            print(f"WARN: 收集到的 activation ({len(sample_acts)}) 少于预测样本数 ({n_pred})。将继续，但可能不一一对应。")
        if n_pred is not None:
            sample_acts = sample_acts[:n_pred]

        # 读取并计算音乐 RMS 如果提供 music_file
        music_rms_z = None
        music_dur = getattr(args, 'music_dur', 60)
        try:
            music_file = args.music_file
            music_start = getattr(args, 'music_start', 0)
            y, sr = librosa.load(music_file, sr=None)
            s0 = int(music_start * sr)
            s1 = min(len(y), s0 + int(music_dur * sr))
            if s0 >= len(y):
                s0 = 0; s1 = len(y)
            y_clip = y[s0:s1]
            frame_length = getattr(args, 'rms_frame_length', 1024)
            hop_length = getattr(args, 'rms_hop_length', 512)
            music_rms = librosa.feature.rms(y=y_clip, frame_length=frame_length, hop_length=hop_length)[0]
            music_rms_z = (music_rms - np.mean(music_rms)) / (np.std(music_rms) + 1e-9)
        except Exception as e:
            print("WARN: 无法读取音乐或计算 RMS，跳过对齐（需要 librosa 且 args.music_file 有效）。 Error:", e)
            music_rms_z = None

        OUT_DIR = os.path.join(args.save_path, 'param_alignment')
        os.makedirs(OUT_DIR, exist_ok=True)
        results = []

        for i, act in enumerate(sample_acts, start=1):
            try:
                act_arr = np.array(act)
                if act_arr.ndim == 3:
                    act_arr = act_arr.squeeze()
                if act_arr.ndim == 2:
                    act_rms = np.sqrt(np.mean(act_arr ** 2, axis=0))
                elif act_arr.ndim == 1:
                    act_rms = np.abs(act_arr)
                else:
                    act_rms = np.sqrt(np.mean(act_arr.reshape(act_arr.shape[0], -1) ** 2, axis=0))

                act_z = (act_rms - np.mean(act_rms)) / (np.std(act_rms) + 1e-9)

                if music_rms_z is None:
                    mse = float('nan'); pearson_r = float('nan')
                    plt.figure(figsize=(8,2))
                    plt.plot(act_z, label='Model response (z)')
                    plt.title(f'Sample {i} — Model response (no music)')
                    plt.xlabel('frame')
                    plt.ylabel('z')
                    plt.tight_layout()
                    plt.savefig(os.path.join(OUT_DIR, f'sample_{i}_resp_nomusic.png'), dpi=300)
                    plt.close()
                    results.append((i, mse, pearson_r))
                    continue

                t_act = np.linspace(0.0, 1.0, num=len(act_z))
                t_mus = np.linspace(0.0, 1.0, num=len(music_rms_z))
                mus_interp = np.interp(t_act, t_mus, music_rms_z)

                mse = float(np.mean((act_z - mus_interp) ** 2))
                try:
                    pearson_r, _ = pearsonr(act_z, mus_interp)
                except Exception:
                    pearson_r = float('nan')

                x_seconds = np.linspace(0, music_dur, num=len(act_z))
                plt.figure(figsize=(10,2))
                plt.plot(x_seconds, act_z, label='Model response (z)')
                plt.plot(x_seconds, mus_interp, label='Music RMS (interp)', alpha=0.9)
                plt.legend(loc='upper right')
                plt.xlabel('Time (s)')
                plt.ylabel('z')
                plt.title(f'Sample {i} — MSE={mse:.6e}, Pearson={pearson_r:.3f}')
                plt.tight_layout()
                plt.savefig(os.path.join(OUT_DIR, f'sample_{i}_resp_vs_music.png'), dpi=300)
                plt.close()

                results.append((i, mse, pearson_r))
            except Exception as e:
                print(f"WARN: 处理 sample {i} activation 失败：", e)
                results.append((i, float('nan'), float('nan')))

        # 写入对齐结果文件
        out_txt = os.path.join(OUT_DIR, f'param_activation_alignment_SUB{subject}_FOLD{fold}.txt')
        with open(out_txt, 'a', encoding='utf-8') as wf:
            wf.write(f"=== SUBJECT {subject} FOLD {fold} — MUSIC {os.path.basename(args.music_file)} ===\n")
            for i, mse, pr in results:
                wf.write(f"Sample {i:03d}: MSE={mse:.6e}, Pearson={pr:.4f}\n")
            wf.write("\n")

    # 返回原来值（不变）
    return acc, pred_test, act_test




def combine_train(args, data_train, label_train, subject, fold, target_acc):
    """
    Second-stage training (fine-tuning) until target accuracy.
    """
    save_name = f'_sub{subject}_fold{fold}'
    set_up(args)
    seed_all(args.random_seed)

    train_loader = get_dataloader(data_train, label_train, args.batch_size)
    model = get_model(args)
    if CUDA:
        model = model.cuda()
    # Load initial model
    model.load_state_dict(torch.load(args.load_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate * 1e-1)
    loss_fn = LabelSmoothing(args.LS_rate) if args.LS else nn.CrossEntropyLoss()

    trlog = {
        'args': vars(args),
        'train_loss': [],
        'train_acc': [],
        'max_acc': 0.0,
        'F1': 0.0
    }

    timer = Timer()

    train_res_file = os.path.join(args.save_path, 'train_result.txt')

    for epoch in range(1, args.max_epoch_cmb + 1):
        loss_cmb, pred_cmb, act_cmb = train_one_epoch(
            data_loader=train_loader, net=model, loss_fn=loss_fn, optimizer=optimizer)
        acc_cmb, f1_cmb, _ = get_metrics(y_pred=pred_cmb, y_true=act_cmb)
        print(f'Stage 2 : epoch {epoch}, for train set loss={loss_cmb:.4f} acc={acc_cmb:.4f} f1={f1_cmb:.4f}')
        with open(train_res_file, 'a', encoding='utf-8') as f:
            f.write(f"SUB:{subject} FOLD:{fold}的训练集周期epoch {epoch}, for train set "
                    f"loss={loss_cmb:.4f} acc={acc_cmb:.4f} f1={f1_cmb:.4f}\n")

        # Early stopping or target reached
        if acc_cmb >= target_acc or epoch == args.max_epoch_cmb:
            print('early stopping!')
            # Save final model for inference
            final_name = 'final_model.pth'
            torch.save(model.state_dict(), os.path.join(args.save_path, final_name))
            # Save reproduce model
            model_name = f'sub{subject}_fold{fold}.pth'
            data_type = f'model_{args.data_format}_{args.label_type}'
            experiment_setting = f'T_{args.T}_pool_{args.pool}'
            save_dir = os.path.join(args.save_path, experiment_setting, data_type)
            ensure_path(save_dir)
            torch.save(model.state_dict(), os.path.join(save_dir, model_name))
            break

        trlog['train_loss'].append(loss_cmb)
        trlog['train_acc'].append(acc_cmb)
        print(f'ETA:{timer.measure()}/{timer.measure(epoch/args.max_epoch_cmb)} SUB:{subject} FOLD:{fold}')

    # Save combine training log
    log_name = 'trlog_comb' + save_name
    experiment_setting = f'T_{args.T}_pool_{args.pool}'
    log_dir = os.path.join(args.save_path, experiment_setting, 'log_train_cmb')
    ensure_path(log_dir)
    torch.save(trlog, os.path.join(log_dir, log_name))
