# ═════════════════════════════════════════════════════════════
# WEEK 3 — LoRA fine-tuning of the image encoder
# ═════════════════════════════════════════════════════════════
class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank update: W x + (B A x) * alpha/r."""

    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        dev, dt = base.weight.device, base.weight.dtype
        self.lora_A = nn.Linear(base.in_features, r, bias=False, device=dev, dtype=dt)
        self.lora_B = nn.Linear(r, base.out_features, bias=False, device=dev, dtype=dt)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)   # adapter starts as a no-op
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scale


def resolve_lora_blocks(sam_model, last_n=None):
    """Indices of the Hiera blocks to adapt: the last `last_n`, or all if None."""
    n = len(sam_model.image_encoder.trunk.blocks)
    if last_n is None or last_n >= n:
        return list(range(n))
    return list(range(n - last_n, n))


def inject_lora(sam_model, r=LORA_R, alpha=LORA_ALPHA, blocks=None):
    """Freeze the whole model, then add LoRA to the qkv projection of the given
    Hiera blocks (all blocks if `blocks` is None).

    Adapting only the later blocks makes each step cheaper (~15% measured on
    Hiera-S at 1024): the frozen early blocks see no trainable input, so
    autograd builds no graph through them and backward stops at the first
    adapted block.

    Works on both SAM2Base (from build_sam2) and SAM2VideoPredictor, since the
    latter subclasses SAM2Base and shares attribute names — so LoRA keys saved
    from training load directly into the video predictor.
    Returns the list of trainable (LoRA) parameters.
    """
    for p in sam_model.parameters():
        p.requires_grad = False

    all_blocks = sam_model.image_encoder.trunk.blocks
    idxs = list(range(len(all_blocks))) if blocks is None else list(blocks)
    n_wrapped = 0
    for i in idxs:
        blk = all_blocks[i]
        if not isinstance(blk.attn.qkv, LoRALinear):
            blk.attn.qkv = LoRALinear(blk.attn.qkv, r=r, alpha=alpha)
            n_wrapped += 1

    trainable = [p for n, p in sam_model.named_parameters() if "lora_" in n]
    for p in trainable:
        p.requires_grad = True

    total = sum(p.numel() for p in sam_model.parameters())
    n_train = sum(p.numel() for p in trainable)
    print(f"LoRA injected into {n_wrapped}/{len(all_blocks)} attention blocks "
          f"(blocks {idxs[0]}–{idxs[-1]}) — trainable params: "
          f"{n_train:,} / {total:,} ({100 * n_train / total:.2f}%)")
    return trainable


def lora_state_dict(sam_model):
    """Only the LoRA adapter weights (a few MB instead of the full model)."""
    return {k: v.detach().cpu() for k, v in sam_model.state_dict().items() if "lora_" in k}


def load_lora_weights(sam_model, lora_sd):
    """Load adapter weights into a model that already has LoRA injected.
    Raises if any LoRA key is missing or unexpected (instead of silently ignoring)."""
    missing, unexpected = sam_model.load_state_dict(lora_sd, strict=False)
    missing_lora = [k for k in missing if "lora_" in k]
    if unexpected or missing_lora:
        raise RuntimeError(
            f"LoRA weights do not match this model: {len(missing_lora)} LoRA keys missing, "
            f"{len(unexpected)} unexpected. First missing: {missing_lora[:3]}  "
            f"First unexpected: {unexpected[:3]}")


class DiceBCELoss(nn.Module):
    """BCE + soft Dice, Dice computed per mask then averaged (so small organs
    like the adrenals count as much as the liver)."""

    def forward(self, logits, targets, smooth=1.0):
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        probs = torch.sigmoid(logits).flatten(1)
        targets = targets.flatten(1)
        inter = (probs * targets).sum(1)
        dice = 1 - (2.0 * inter + smooth) / (probs.sum(1) + targets.sum(1) + smooth)
        return bce + dice.mean()


def latest_lora_ckpt(ckpt_dir):
    """Path of the highest-epoch LoRA checkpoint in ckpt_dir, or None."""
    ckpts = sorted(Path(ckpt_dir).glob("sam2_btcv_lora_epoch_*.pt"))
    return ckpts[-1] if ckpts else None


def _check_resume_compatible(ck, r, alpha, blocks, path):
    saved = (ck.get("lora_r"), ck.get("lora_alpha"), ck.get("lora_blocks"))
    wanted = (r, alpha, list(blocks))
    if saved != wanted:
        raise RuntimeError(
            f"{path} was trained with (r, alpha, blocks) = {saved}, but the current "
            f"config is {wanted}. Point CKPT_OUT_DIR at a new folder (or move the old "
            f"checkpoints out) to start a fresh run.")


def train_image_encoder(sam_model, dataset, epochs=10, batch_size=4, lr=1e-4,
                        device="cuda", ckpt_dir=None, r=LORA_R, alpha=LORA_ALPHA,
                        lora_last_n_blocks=None, time_budget_hours=None,
                        num_workers=None):
    """Train LoRA adapters in the image encoder; everything else stays frozen.

    Saves adapters + optimizer state to ckpt_dir every epoch and resumes from
    the latest one. bf16 autocast where supported (A100/L4); fp16 + GradScaler
    otherwise (T4).
    """
    t_start = time.time()
    sam_model.to(device)
    blocks = resolve_lora_blocks(sam_model, lora_last_n_blocks)
    trainable = inject_lora(sam_model, r=r, alpha=alpha, blocks=blocks)
    sam_model.train()
    image_size = sam_model.image_size

    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 2)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, collate_fn=collate_slices,
                            pin_memory=(device == "cuda"), drop_last=True,
                            persistent_workers=num_workers > 0)
    optimizer = AdamW(trainable, lr=lr, weight_decay=1e-4)
    criterion = DiceBCELoss()

    # Precision: bf16 where supported (A100/L4), else fp16 + loss scaling (T4)
    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    print(f"Mixed precision: {amp_dtype if use_amp else 'off'} | "
          f"{len(dataset)} slices, {len(dataloader)} steps/epoch, batch {batch_size}")

    # Resume after a Colab disconnect
    start_epoch = 0
    if ckpt_dir is not None and (last := latest_lora_ckpt(ckpt_dir)) is not None:
        ck = torch.load(last, map_location=device)
        _check_resume_compatible(ck, r, alpha, blocks, last)
        load_lora_weights(sam_model, ck["lora_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        start_epoch = ck["epoch"]
        print(f"Resumed from {last} (epoch {start_epoch}/{epochs})")
        if start_epoch >= epochs:
            print("Training already complete — nothing to do.")
            return sam_model

    budget_s = time_budget_hours * 3600 if time_budget_hours else None
    epoch_durations = []

    for epoch in range(start_epoch, epochs):
        # Stop before starting an epoch that would overrun the budget
        if budget_s is not None and epoch_durations:
            remaining = budget_s - (time.time() - t_start)
            if max(epoch_durations[-2:]) > remaining:
                print(f"Time budget: {remaining / 60:.0f} min left, but an epoch takes "
                      f"~{epoch_durations[-1] / 60:.0f} min — stopping after epoch {epoch}. "
                      f"The epoch-{epoch} checkpoint is your final model.")
                break

        t_epoch = time.time()
        data_wait = 0.0
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        t_fetch = time.time()

        for imgs, masks, boxes, img_idx in progress_bar:
            data_wait += time.time() - t_fetch

            imgs = gpu_prepare_images(imgs, image_size, device)             # (B, 3, S, S)
            masks = masks.to(device, non_blocking=True).unsqueeze(1)        # (N, 1, S/4, S/4)
            boxes = boxes.to(device, non_blocking=True)                     # (N, 4)
            img_idx = img_idx.to(device, non_blocking=True)                 # (N,)
            B = imgs.shape[0]

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                backbone_out = sam_model.forward_image(imgs)
                _, vision_feats, _, feat_sizes = sam_model._prepare_backbone_features(backbone_out)
                if sam_model.directly_add_no_mem_embed:
                    vision_feats[-1] = vision_feats[-1] + sam_model.no_mem_embed

                # vision_feats[i]: (H_i*W_i, B, C), high-res -> low-res; sizes from feat_sizes
                feats = [f.permute(1, 2, 0).reshape(B, -1, h, w)
                         for f, (h, w) in zip(vision_feats, feat_sizes)]
                # One encoding per image -> one row per prompt (cheap gather)
                image_embeddings = feats[-1][img_idx]
                high_res_features = [f[img_idx] for f in feats[:-1]]

                with torch.no_grad():
                    sparse_embeddings, dense_embeddings = sam_model.sam_prompt_encoder(
                        points=None, boxes=boxes.unsqueeze(1), masks=None,
                    )

                low_res_masks, _, _, _ = sam_model.sam_mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=sam_model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=high_res_features,
                )

            loss = criterion(low_res_masks.float(), masks)                  # fp32, 256x256
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "prompts": len(boxes)})
            t_fetch = time.time()

        dur = time.time() - t_epoch
        epoch_durations.append(dur)
        avg = epoch_loss / max(1, len(dataloader))
        print(f"Epoch {epoch + 1} | loss {avg:.4f} | {dur / 60:.1f} min "
              f"(waiting for data {100 * data_wait / dur:.0f}%)")
        if data_wait / dur > 0.3:
            print("  NOTE: >30% of time spent waiting for data — the DataLoader, not "
                  "the GPU, is the bottleneck. Keep PREPROCESSED_ROOT on local /content "
                  "disk, not Drive.")

        if ckpt_dir is not None:
            Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
            out = Path(ckpt_dir) / f"sam2_btcv_lora_epoch_{epoch + 1:03d}.pt"
            tmp = out.with_suffix(".tmp")
            torch.save({
                "epoch": epoch + 1,
                "loss": avg,
                "lora_r": r,
                "lora_alpha": alpha,
                "lora_blocks": list(blocks),
                "lora_state_dict": lora_state_dict(sam_model),
                "optimizer_state_dict": optimizer.state_dict(),
            }, tmp)
            tmp.replace(out)   # atomic: a disconnect mid-save can't leave a corrupt ckpt
            print(f"  saved LoRA checkpoint -> {out}")

        if epoch == start_epoch and epochs - epoch > 1:
            eta = dur * (epochs - epoch - 1)
            print(f"  estimated time for remaining {epochs - epoch - 1} epochs: "
                  f"{eta / 3600:.1f} h")

        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"Training finished in {(time.time() - t_start) / 3600:.2f} h")
    return sam_model


# ═════════════════════════════════════════════════════════════
# WEEK 4 — metrics
# ═════════════════════════════════════════════════════════════
def dice_score(pred3d, gt3d):
    """Volumetric Dice. Both empty -> 1.0; one empty -> 0.0."""
    pred = np.asarray(pred3d).astype(bool)
    gt = np.asarray(gt3d).astype(bool)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    ps, gs = pred.sum(), gt.sum()
    if ps == 0 and gs == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / (ps + gs))


def _surface_voxels(mask):
    if not np.any(mask):
        return mask
    eroded = binary_erosion(mask, iterations=1, border_value=0)
    return np.logical_and(mask, np.logical_not(eroded))


def hd95(pred3d, gt3d, spacing=(1.0, 1.0, 1.0)):
    """Symmetric 95th-percentile Hausdorff distance in mm.
    Both empty -> 0.0; exactly one empty -> inf (a total miss)."""
    pred = np.asarray(pred3d).astype(bool)
    gt = np.asarray(gt3d).astype(bool)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    pe, ge = not pred.any(), not gt.any()
    if pe and ge:
        return 0.0
    if pe or ge:
        return float("inf")

    ps, gs = _surface_voxels(pred), _surface_voxels(gt)
    dt_gt = distance_transform_edt(np.logical_not(gs), sampling=spacing)
    dt_pred = distance_transform_edt(np.logical_not(ps), sampling=spacing)
    d = np.concatenate([dt_gt[ps], dt_pred[gs]])
    return float(np.percentile(d, 95))


# ═════════════════════════════════════════════════════════════
# WEEK 4 — evaluation
# ═════════════════════════════════════════════════════════════
def _load_predictor(cfg, lora_path=None):
    """Build a SAM 2 video predictor; optionally load fine-tuned weights.

    - LoRA checkpoint (has 'lora_state_dict'): adapters are injected into the
      predictor first, then loaded, and every LoRA key is verified to match.
    - Legacy checkpoint (full state dict, possibly wrapped): loaded as before.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: running on CPU — SAM 2 video propagation will be very slow.")

    predictor = build_predictor(cfg["model"]["cfg"], cfg["model"]["ckpt"], device=device)

    if lora_path is None:
        return predictor

    raw = torch.load(lora_path, map_location=device)

    if isinstance(raw, dict) and "lora_state_dict" in raw:
        r = raw.get("lora_r", LORA_R)
        alpha = raw.get("lora_alpha", LORA_ALPHA)
        blocks = raw.get("lora_blocks")   # None (older ckpts) = all blocks
        inject_lora(predictor, r=r, alpha=alpha, blocks=blocks)
        lora_sd = raw["lora_state_dict"]
        load_lora_weights(predictor, lora_sd)
        print(f"  [LoRA] loaded {len(lora_sd)} adapter tensors (r={r}, alpha={alpha})")
    else:
        if isinstance(raw, dict) and "model_state_dict" in raw:
            state_dict = raw["model_state_dict"]
        elif isinstance(raw, dict) and "model" in raw:
            state_dict = raw["model"]
        elif isinstance(raw, dict) and "state_dict" in raw:
            state_dict = raw["state_dict"]
        else:
            state_dict = raw
        missing, unexpected = predictor.load_state_dict(state_dict, strict=False)
        if len(missing) > 50 or len(unexpected) > 50:
            print(f"  WARNING: large load_state_dict mismatch (missing={len(missing)}, "
                  f"unexpected={len(unexpected)}). First missing: {missing[:5]}")
        else:
            print(f"  [legacy full ckpt] missing={len(missing)} unexpected={len(unexpected)}")

    predictor.eval()
    return predictor


def _frames_dir_for(cfg, case):
    return Path(cfg["paths"]["frames_root"]) / case


def _ensure_frames(cfg, case, vol_u8):
    """Write .jpg frames for a case if not already cached (SAM 2's video loader
    reads JPEGs and applies ImageNet normalization itself)."""
    frames_dir = _frames_dir_for(cfg, case)
    num_expected = vol_u8.shape[2]

    if frames_dir.exists():
        if len(list(frames_dir.glob("*.jpg"))) == num_expected:
            return frames_dir
        for f in frames_dir.glob("*"):
            f.unlink()
    else:
        frames_dir.mkdir(parents=True, exist_ok=True)

    for z in range(num_expected):
        Image.fromarray(to_rgb(vol_u8[:, :, z])).save(frames_dir / f"{z:05d}.jpg", quality=95)
    return frames_dir


def _run_case(predictor, cfg, case, organ_id):
    """Return (pred3d, gt3d, spacing), or None if the organ is absent in this case."""
    vol, _, spacing = load_volume(str(Path(cfg["paths"]["images"]) / f"{case}.nii"))
    lbl, _, _ = load_volume(str(Path(cfg["paths"]["labels"]) / f"{case.replace('img', 'label')}.nii"))
    vol_u8 = apply_hu_window(vol)

    organ_mask3d = (lbl == organ_id)
    if not np.any(organ_mask3d):
        return None

    start_z = best_start_slice(lbl, organ_id)
    bbox = bbox_from_mask(organ_mask3d[:, :, start_z].astype(np.uint8), pad=4)

    frames_dir = _ensure_frames(cfg, case, vol_u8)
    state = init_state(predictor, frames_dir)
    pred3d = propagate_bidirectional(predictor, state, start_z, bbox,
                                     target_hw=vol_u8.shape[:2])

    if hasattr(predictor, "reset_state"):
        predictor.reset_state(state)
    del state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pred3d.astype(np.uint8), organ_mask3d.astype(np.uint8), spacing


def evaluate_organ(cfg, organ_id, lora_path=None):
    """Zero-shot (+ LoRA if lora_path) inference and metrics on every val case."""
    zs_predictor = _load_predictor(cfg, lora_path=None)
    lora_predictor = _load_predictor(cfg, lora_path=lora_path) if lora_path else None

    rows = []
    for case in cfg["split"]["val_cases"]:
        print(f"  [{case}] zero-shot propagation...")
        zs = _run_case(zs_predictor, cfg, case, organ_id)
        if zs is None:
            print(f"  [{case}] organ {organ_id} not present — skipping.")
            continue
        pred_zs, gt3d, spacing = zs

        row = {
            "case": case,
            "dsc_zs": dice_score(pred_zs, gt3d),
            "hd95_zs": hd95(pred_zs, gt3d, spacing=spacing),
            "dsc_lora": None,
            "hd95_lora": None,
        }

        if lora_predictor is not None:
            print(f"  [{case}] LoRA propagation...")
            lr_res = _run_case(lora_predictor, cfg, case, organ_id)
            if lr_res is not None:
                pred_l, gt_l, sp_l = lr_res
                row["dsc_lora"] = dice_score(pred_l, gt_l)
                row["hd95_lora"] = hd95(pred_l, gt_l, spacing=sp_l)

        rows.append(row)

    return rows


def _mean_std(values):
    clean = [v for v in values if v is not None and np.isfinite(v)]
    if not clean:
        return float("nan"), float("nan")
    return float(np.mean(clean)), float(np.std(clean))


def print_table(rows, organ_id):
    """Print the DSC / HD95 summary table and return the aggregated stats."""
    dsc_zs_mean, dsc_zs_std = _mean_std([r["dsc_zs"] for r in rows])
    hd_zs_mean, hd_zs_std = _mean_std([r["hd95_zs"] for r in rows])
    has_lora = any(r["dsc_lora"] is not None for r in rows)
    if has_lora:
        dsc_l_mean, dsc_l_std = _mean_std([r["dsc_lora"] for r in rows])
        hd_l_mean, hd_l_std = _mean_std([r["hd95_lora"] for r in rows])

    name = ORGAN_IDS.get(organ_id, str(organ_id))
    print("=" * 60)
    print(f"  Organ {organ_id} ({name})  |  {len(rows)} val cases")
    print("=" * 60)
    if has_lora:
        print(f"  {'Metric':<14} {'Zero-shot':<20} {'LoRA'}")
        print("  " + "-" * 50)
        print(f"  {'DSC':<14} {dsc_zs_mean:.3f} \u00b1 {dsc_zs_std:.3f}"
              f"{'':<8}{dsc_l_mean:.3f} \u00b1 {dsc_l_std:.3f}")
        print(f"  {'HD95 (mm)':<14} {hd_zs_mean:.1f}   \u00b1 {hd_zs_std:.1f}"
              f"{'':<9}{hd_l_mean:.1f}   \u00b1 {hd_l_std:.1f}")
    else:
        print(f"  {'Metric':<14} {'Zero-shot'}")
        print("  " + "-" * 30)
        print(f"  {'DSC':<14} {dsc_zs_mean:.3f} \u00b1 {dsc_zs_std:.3f}")
        print(f"  {'HD95 (mm)':<14} {hd_zs_mean:.1f}   \u00b1 {hd_zs_std:.1f}")
    print("=" * 60)

    summary = {
        "organ_id": organ_id, "n_cases": len(rows),
        "dsc_zs_mean": dsc_zs_mean, "dsc_zs_std": dsc_zs_std,
        "hd95_zs_mean": hd_zs_mean, "hd95_zs_std": hd_zs_std,
    }
    if has_lora:
        summary.update({
            "dsc_lora_mean": dsc_l_mean, "dsc_lora_std": dsc_l_std,
            "hd95_lora_mean": hd_l_mean, "hd95_lora_std": hd_l_std,
        })
    return summary


# ═════════════════════════════════════════════════════════════
# DRIVERS
# ═════════════════════════════════════════════════════════════
def main_train():
    from sam2.build_sam import build_sam2

    cases = discover_cases(TRAIN_IMAGE_DIR)
    preprocess_slices(cases, TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR,
                      PREPROCESSED_ROOT, ORGAN_IDS, slice_stride=SLICE_STRIDE)

    dataset = BTCVSliceDataset(PREPROCESSED_ROOT, ORGAN_IDS, image_size=IMAGE_SIZE,
                               max_prompts=MAX_PROMPTS_PER_SLICE)

    sam_model = build_sam2(MODEL_CFG, BASE_CKPT, device=DEVICE)
    train_image_encoder(sam_model, dataset, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        lr=LR, device=DEVICE, ckpt_dir=CKPT_OUT_DIR,
                        r=LORA_R, alpha=LORA_ALPHA,
                        lora_last_n_blocks=LORA_LAST_N_BLOCKS,
                        time_budget_hours=TIME_BUDGET_HOURS)


def build_eval_cfg():
    return {
        "paths": {"images": VAL_IMAGE_DIR, "labels": VAL_LABEL_DIR,
                  "frames_root": FRAMES_ROOT},
        "model": {"cfg": MODEL_CFG, "ckpt": BASE_CKPT},
        "split": {"val_cases": VAL_CASES},
    }


def run_single_organ(cfg, organ_id, finetuned_ckpt, output_dir):
    print(f"\nEvaluating organ={organ_id}  val cases={cfg['split']['val_cases']}\n")
    rows = evaluate_organ(cfg, organ_id, lora_path=finetuned_ckpt)
    if not rows:
        print(f"No results for organ {organ_id} — check that it appears in the val cases.")
        return None

    summary = print_table(rows, organ_id)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"eval_organ{organ_id}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Per-case CSV saved -> {csv_path}")
    return summary


def main_eval():
    ckpt = FINETUNED_CKPT or latest_lora_ckpt(CKPT_OUT_DIR)
    if ckpt is None:
        print(f"No LoRA checkpoint found in {CKPT_OUT_DIR} — running zero-shot only.")
    else:
        print(f"Evaluating with fine-tuned checkpoint: {ckpt}")
    cfg = build_eval_cfg()
    summaries = {}
    for organ_id in ORGANS_TO_EVAL:
        s = run_single_organ(cfg, organ_id, str(ckpt) if ckpt else None, EVAL_OUT_DIR)
        if s is not None:
            summaries[organ_id] = s
    return summaries


def main():
    colab_setup()
    if RUN_TRAIN:
        main_train()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if RUN_EVAL:
        return main_eval()


if __name__ == "__main__":
    main()
