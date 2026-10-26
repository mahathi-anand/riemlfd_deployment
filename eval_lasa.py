"""Evaluate a RiemannianLfD LASA latent NODE (N, Angle, ...) on an FR3 through CRISP.

The LASA shapes live in a normalised 2D frame (x, y in [-1, 1], z = 0) with a unit
quaternion (w, x, y, z) that converges to identity at the goal. They are mapped to the robot as
    position:     p = center + scale * (u_x * e1 + u_y * e2), (e1, e2) spanning `plane`
    orientation:  R = R_ref * R_lasa, R_ref the end-effector orientation after homing
and every control tick runs the same loop as deploy_node.py:
    z --NODE step--> z' --decode--> LASA pose --map--> target, clamped to max_step / max_rot.
Latent and hidden sizes are read from the NODE checkpoint. At the end the demonstrations, the
commanded and the measured path are plotted in the LASA frame and the samples saved as .npz.

The model is picked on the command line (--shape, --beta_scale, --suffix); every other setting
is in config/lasa_eval.yaml: `control_loop` (rate, look-ahead, RK4 substeps, timeout) for every shape,
and each shape's own section (e.g. `Sine:`) for its start and goal, mapping, stop tolerance, safety,
output and dry_run. A shape can only be run if it has a section.

Run (sim or real robot already up):
    pixi run python eval_lasa.py --shape N
    pixi run python eval_lasa.py --shape Angle --beta_scale 5
"""
import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from scipy.io import loadmat
from scipy.spatial.transform import Rotation, Slerp

from lfd_models import load_node, load_vae

HERE = os.path.dirname(os.path.abspath(__file__))
# copied from RiemannianLfD: models/vae/lasa/ -> vae/, models/node/lasa/ -> node/, datasets/lasa/*_UQ.mat -> here
BUNDLES = os.path.join(HERE, "bundles", "lasa")
N_DEMOS = 7  # LASA demonstrations per shape

PLANES = {"yz": ([0, 1, 0], [0, 0, 1]), "xy": ([1, 0, 0], [0, 1, 0]), "xz": ([1, 0, 0], [0, 0, 1])}
CONFIG = os.path.join(HERE, "config", "lasa_eval.yaml")  # `control_loop` for all shapes, then one section per shape
# key -> type of its value (None allowed for the NULLABLE keys)
LOOP_KEYS = {"rate": float, "lookahead": float, "substeps": int, "timeout": float}
SHAPE_KEYS = {"start_demo": int, "start_uv": list, "goal_demo": int,
              "plane": str, "center": list, "scale": float, "no_orientation": bool, "goal_tol": float,
              "max_step": float, "max_rot": float, "resync": float, "off_data_margin": float,
              "off_data_hold": bool, "workspace_margin": float, "params": str, "output": str, "dry_run": bool}
KEYS = {**LOOP_KEYS, **SHAPE_KEYS}
NULLABLE = {"start_uv", "params", "output"}


def read_config(config_path):
    """The config's `control_loop` section and its shape sections (every other top-level key)."""
    with open(config_path) as f:
        full = yaml.safe_load(f) or {}
    if "control_loop" not in full:
        sys.exit(f"{config_path}: no 'control_loop' section")
    shapes = {k: v or {} for k, v in full.items() if k != "control_loop"}
    if not shapes:
        sys.exit(f"{config_path}: no shape sections")
    return full["control_loop"] or {}, shapes


def load_config(config_path, loop, section, shape):
    """The `control_loop` settings plus the shape's section, checked for missing, unknown and mistyped keys."""
    path = f"{config_path} ('control_loop' + '{shape}')"
    for name, part, keys in [("control_loop", loop, LOOP_KEYS), (shape, section, SHAPE_KEYS)]:
        missing, unknown = keys.keys() - part.keys(), part.keys() - keys.keys()
        if missing or unknown:
            sys.exit(f"{config_path}, section '{name}': " + "; ".join(filter(None, [
                missing and f"missing {sorted(missing)}", unknown and f"unknown {sorted(unknown)}"])))
    cfg = {**loop, **section}
    for key, typ in KEYS.items():
        v = cfg[key]
        if v is None and key in NULLABLE:
            continue
        if typ is float and isinstance(v, int) and not isinstance(v, bool):
            v = cfg[key] = float(v)
        if not isinstance(v, typ) or (typ is int and isinstance(v, bool)):
            sys.exit(f"{path}: '{key}' must be {typ.__name__}, got {v!r}")
    if cfg["plane"] not in PLANES:
        sys.exit(f"{path}: 'plane' must be one of {list(PLANES)}, got {cfg['plane']!r}")
    for key, n in [("center", 3), ("start_uv", 2)]:
        if cfg[key] is not None and (len(cfg[key]) != n or not all(isinstance(x, (int, float)) for x in cfg[key])):
            sys.exit(f"{path}: '{key}' must be a list of {n} numbers, got {cfg[key]!r}")
    for key in ["params", "output"]:  
        if cfg[key] is not None:
            cfg[key] = os.path.join(HERE, os.path.expanduser(cfg[key]))
    return cfg


loop_cfg, shape_cfgs = read_config(CONFIG)
p = argparse.ArgumentParser(description=f"Evaluate a LASA NODE; every other setting is in {CONFIG} "
                                        "('control_loop' plus the shape's own section).")
p.add_argument("--shape", required=True, choices=list(shape_cfgs), help="a shape section of the config")
p.add_argument("--beta_scale", default="10", help="part of the model file names (N, Angle, Sine, Leaf_1: 5, 10)")
p.add_argument("--suffix", default="energy_imitation", help="NODE training variant: energy_imitation | energy_only | imitation_only")
args = p.parse_args()
a = SimpleNamespace(**vars(args), **load_config(CONFIG, loop_cfg, shape_cfgs[args.shape], args.shape))
print(f"settings from {CONFIG}: 'control_loop' + '{args.shape}'")

# ---------------------------------------------------------------- models
out = os.path.abspath(a.output or os.path.join(HERE, "results", f"lasa_{a.shape}_beta_{a.beta_scale}"))
shape = f"{a.shape}-Shape"
vae_path = os.path.join(BUNDLES, "vae", f"VAE_{shape}_beta_scale_{a.beta_scale}.pt")
node_path = os.path.join(BUNDLES, "node", f"NODE_lasa_{shape}_beta_scale_{a.beta_scale}_RiemannianMSE_EGI_{a.suffix}.pth")
mat_path = os.path.join(BUNDLES, f"{shape}_UQ.mat")
for path in [vae_path, node_path, mat_path]:
    if not os.path.exists(path):
        folder = os.path.dirname(path)
        sys.exit(f"missing {path}\nin {folder}: " + (", ".join(sorted(os.listdir(folder))) or "nothing"))


def lasa_points(path):
    """The VAE training points, built as RiemannianLfD's get_demonstrations_paths_newVAE does:
    x, y normalised by one shared isotropic scale over all demos, z kept, quaternion (w, x, y, z),
    and a [+q, -q] copy of every demo (the double cover the VAE was trained on)."""
    d = loadmat(path)["demoUQ"]
    pos = [d[0, i]["tsPos"][0, 0].T for i in range(N_DEMOS)]
    quat = [d[0, i]["quat"][0, 0].T for i in range(N_DEMOS)]
    xy = np.vstack([p[:, :2] for p in pos])
    lo, hi = xy.min(0), xy.max(0)
    c, s = (lo + hi) / 2, (hi - lo).max() / 2
    trajs = []
    for p, q in zip(pos, quat):
        pn = np.hstack([(p[:, :2] - c) / s, p[:, 2:3]])
        trajs += [np.hstack([pn, q]), np.hstack([pn, -q])]
    return torch.from_numpy(np.vstack(trajs)).float()


X = lasa_points(mat_path)  # [demos * 2 * T, 7], +q/-q per demo, (w, x, y, z)
T = len(X) // (2 * N_DEMOS)
vae, node = load_vae(vae_path), load_node(node_path)
latent_dim = node.ode_func.net[-1].out_features
print(f"{shape}, beta_scale {a.beta_scale}: NODE latent_dim {latent_dim}, hidden_dim {node.ode_func.net[0].out_features}")


@torch.no_grad()
def encode(u7):
    return vae.encode(torch.as_tensor(np.asarray(u7)[None], dtype=torch.float32))


@torch.no_grad()
def decode(z):
    pos, q = vae.decode(z.reshape(-1, latent_dim))
    return pos[0].numpy(), q[0].numpy()


@torch.no_grad()
def integrate(z, zg, h):
    dt = h / a.substeps
    f = lambda z_: node.ode_func(None, z_, zg)
    for _ in range(a.substeps):  # RK4, same scheme as training
        k1 = f(z); k2 = f(z + dt / 2 * k1); k3 = f(z + dt / 2 * k2); k4 = f(z + dt * k3)
        z = z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return z


@torch.no_grad()
def logdetG(z):
    return torch.logdet(vae.metric(z.reshape(-1, latent_dim)))


with torch.no_grad():
    ld_data = logdetG(vae.encode(X[::10])) #Encode demonstrations to get logdetG values
off_thr = ld_data[torch.isfinite(ld_data)].max().item() + a.off_data_margin #Compute off-data threshold from demos
# far off the data G is so ill-conditioned that its float32 log det comes out NaN: that is off-data too
off_data = lambda z: not logdetG(z).item() <= off_thr
demo = lambda i: X[2 * T * i: 2 * T * i + T].numpy()  # +q copy of demo i
for key in ["start_demo", "goal_demo"]:
    if not 0 <= getattr(a, key) < N_DEMOS:
        sys.exit(f"{CONFIG}: '{key}' must be 0-{N_DEMOS - 1}, got {getattr(a, key)}")

# ---------------------------------------------------------------- LASA frame <-> robot base frame
E = np.array(PLANES[a.plane], dtype=float).T  # 3x2, columns e1, e2
C = np.array(a.center)
R_ref = Rotation.identity()  # replaced by the homed orientation on the robot
wxyz = lambda r: np.roll(r.as_quat(), 1)
from_wxyz = lambda q: Rotation.from_quat(np.roll(q, -1))


def to_robot(u_pos, u_quat):
    """Robot pose of a LASA pose; x, y are clipped to the shapes' box [-1, 1] plus workspace_margin."""
    lim = 1.0 + a.workspace_margin
    rot = R_ref if a.no_orientation else R_ref * from_wxyz(u_quat)
    return C + a.scale * E @ np.clip(u_pos[:2], -lim, lim), rot


def to_lasa(pos, rot, ref_quat):
    """7D LASA point of a robot pose; with --no_orientation the quaternion is taken from ref_quat."""
    uq = ref_quat if a.no_orientation else wxyz(R_ref.inv() * rot)
    if np.dot(uq, ref_quat) < 0:  # pick the cover the demonstrations use
        uq = -uq
    return np.concatenate([E.T @ (pos - C) / a.scale, [0.0], uq])


def step(z, zg):
    """Advance z by --lookahead NODE time, shrunk so the decoded robot position moves at most max_step."""
    p0 = a.scale * decode(z)[0][:2]
    z1 = integrate(z, zg, a.lookahead)
    d = np.linalg.norm(a.scale * decode(z1)[0][:2] - p0)
    if d > a.max_step:
        z1 = integrate(z, zg, a.lookahead * a.max_step / d)
    return z1


def clamp(pos, rot, tgt_pos, tgt_rot):
    """Limit the change from (pos, rot) towards (tgt_pos, tgt_rot) to max_step / max_rot."""
    d = tgt_pos - pos
    n = np.linalg.norm(d)
    if n > a.max_step:
        d *= a.max_step / n
    ang = np.degrees((rot.inv() * tgt_rot).magnitude())
    s = 1.0 if ang <= a.max_rot else a.max_rot / ang
    return pos + d, Slerp([0, 1], Rotation.concatenate([rot, tgt_rot]))([s])[0]


goal7 = demo(a.goal_demo)[-1]
zg = encode(goal7)
goal_dec = decode(zg)[0]
if a.start_uv is None:
    start7 = demo(a.start_demo)[0]
else:
    plus = np.vstack([demo(i) for i in range(N_DEMOS)])
    start7 = plus[np.argmin(np.linalg.norm(plus[:, :2] - np.array(a.start_uv), axis=1))].copy()
    start7[:2] = a.start_uv
print(f"start {np.round(start7[:2], 3)}, goal {np.round(goal7[:2], 3)} (decoded {np.round(goal_dec[:2], 3)}) "
      f"in the LASA frame; {a.scale * 2:.2f} m wide in the robot {a.plane} plane at {C}; "
      f"off-data if log det G > {off_thr:.2f}")
if off_data(encode(start7)):
    print("warning: the start is off the data, " +
          ("the robot will hold instead of moving" if a.off_data_hold else "following the NODE anyway (off_data_hold: false)"))


def plot(cmd_u, meas_u, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    for i in range(N_DEMOS):
        ax.plot(*demo(i)[:, :2].T, color="0.75", lw=1, label="demonstrations" if i == 0 else None)
    ax.plot(*np.asarray(cmd_u)[:, :2].T, "--", color="tab:orange", label="commanded")
    if meas_u is not None:
        ax.plot(*np.asarray(meas_u)[:, :2].T, color="tab:blue", label="measured")
    ax.plot(*start7[:2], "o", color="green", label="start")
    ax.plot(*goal7[:2], "X", color="black", ms=9, label="goal")
    ax.set(xlabel="LASA x", ylabel="LASA y", title=title, aspect="equal")
    ax.grid(alpha=0.3)
    ax.legend()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out + ".png", dpi=150)
    print(f"plot saved to {out}.png")


# ---------------------------------------------------------------- dry run (no ROS)
if a.dry_run:
    pos, rot = to_robot(start7[:3], start7[3:])
    z, cmd_u, off = encode(start7), [], False
    for k in range(int(a.timeout * a.rate)):
        if off_data(z):
            if a.off_data_hold:
                print(f"tick {k}: off-data, would hold"); break
            if not off:
                print(f"tick {k}: off-data, continuing (off_data_hold: false)")
            off = True
        elif off:
            print(f"tick {k}: back on the data")
            off = False
        z = step(z, zg)
        tp, tq = to_robot(*decode(z))
        pos, rot = clamp(pos, rot, tp, tq)
        cmd_u.append(to_lasa(pos, rot, goal7[3:]))
        if np.linalg.norm(pos - to_robot(goal_dec, goal7[3:])[0]) < a.goal_tol:
            break
    err = np.linalg.norm(pos - to_robot(goal7[:3], goal7[3:])[0])
    print(f"dry run: {k + 1} ticks ({(k + 1) / a.rate:.1f} s at {a.rate:.0f} Hz), "
          f"final error to the real goal {1000 * err:.1f} mm")
    plot(cmd_u, None, f"{shape} dry run, beta_scale {a.beta_scale}")
    sys.exit(0)

# ---------------------------------------------------------------- robot
from crisp_py.robot import make_robot  # noqa: E402
from crisp_py.utils.geometry import Pose  # noqa: E402

robot = make_robot("fr3")
samples = []  # (t, commanded 7D LASA, measured 7D LASA, commanded xyz, measured xyz)
try:
    robot.wait_until_ready()
    robot.home()
    R_ref = robot.end_effector_pose.orientation
    robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
    if a.params:
        robot.cartesian_controller_parameters_client.load_param_config(file_path=a.params)

    sp, sr = to_robot(start7[:3], start7[3:])
    print(f"moving to the start {np.round(sp, 3)}")
    robot.move_to(pose=Pose(sp, sr), speed=0.05)
    time.sleep(1.5)

    # The demonstrated orientations lie on a thin set (Angle: one rotation axis), so a couple of degrees
    # of orientation tracking error already encodes off-data. Measured poses are therefore re-encoded
    # with their measured position and the last commanded orientation, which is on the data.
    encode_measured = lambda ee, cmd: encode(to_lasa(ee.position, cmd.orientation, goal7[3:]))

    print("running the NODE (Ctrl+C to stop)")
    cmd = Pose(sp.copy(), sr)  # last commanded target, on the data; clamping is relative to it
    z = encode(start7)
    goal_pos = to_robot(goal_dec, goal7[3:])[0]
    t0 = next_tick = time.monotonic()
    k, holding = 0, False
    safe_cmd, safe_z = cmd.copy(), z.clone()  # last commanded target and latent state that were on the data
    while time.monotonic() - t0 < a.timeout:
        tick = time.monotonic()
        ee = robot.end_effector_pose.copy()
        if np.linalg.norm(ee.position - cmd.position) > a.resync:  # pushed away: restart from here
            if not holding:
                print(f"[{tick - t0:5.1f}s] disturbance of {1000 * np.linalg.norm(ee.position - cmd.position):.0f} mm, re-encoding")
            z, cmd = encode_measured(ee, cmd), Pose(ee.position.copy(), cmd.orientation)
        off = off_data(z)
        if off and a.off_data_hold:
            # Return to the last on-data target instead of holding where the robot is: that spot may itself
            # be off the data, and the robot would never resume. The impedance controller pulls the arm back,
            # and the NODE continues from the matching on-data latent state once the arm is within `resync`.
            if not holding:
                print(f"[{tick - t0:5.1f}s] off-data, returning to the last on-data target {np.round(safe_cmd.position, 3)}")
            holding = True
            cmd, z = safe_cmd.copy(), safe_z.clone()
            robot.set_target(pose=cmd)
        else:
            if off and not holding:
                print(f"[{tick - t0:5.1f}s] off-data, continuing (off_data_hold: false)")
            elif not off and holding:
                print(f"[{tick - t0:5.1f}s] back on the data" + (", resuming" if a.off_data_hold else ""))
            holding = off  # with off_data_hold false this only tracks the state for the messages
            if not off:
                safe_cmd, safe_z = cmd.copy(), z.clone()
            z = step(z, zg)
            tp, tq = to_robot(*decode(z))
            cmd = Pose(*clamp(cmd.position, cmd.orientation, tp, tq))
            robot.set_target(pose=cmd)
        samples.append((tick - t0, to_lasa(cmd.position, cmd.orientation, goal7[3:]),
                        to_lasa(ee.position, ee.orientation, goal7[3:]), cmd.position.copy(), ee.position.copy()))
        if k % int(a.rate) == 0:
            print(f"[{tick - t0:5.1f}s] ee {np.round(ee.position, 3)}  to goal {1000 * np.linalg.norm(ee.position - goal_pos):6.1f} mm")
        if np.linalg.norm(ee.position - goal_pos) < a.goal_tol and np.linalg.norm(robot.end_effector_twist.linear) < 0.01:
            print(f"reached the goal: {1000 * np.linalg.norm(ee.position - to_robot(goal7[:3], goal7[3:])[0]):.1f} mm "
                  f"from the real goal after {tick - t0:.1f} s")
            break
        k += 1
        next_tick += 1.0 / a.rate
        time.sleep(max(0.0, next_tick - time.monotonic()))
except KeyboardInterrupt:
    pass
finally:
    if samples:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez(out + ".npz", time_s=np.array([s[0] for s in samples]),
                 cmd_lasa=np.array([s[1] for s in samples]), meas_lasa=np.array([s[2] for s in samples]),
                 cmd_pos_m=np.array([s[3] for s in samples]), meas_pos_m=np.array([s[4] for s in samples]))
        print(f"samples saved to {out}.npz")
    if robot.end_effector_pose is not None:
        robot.set_target(pose=robot.end_effector_pose)  # hold where we are
    robot.shutdown()
if samples:
    plot([s[1] for s in samples], [s[2] for s in samples], f"{shape} on the FR3, beta_scale {a.beta_scale}")
