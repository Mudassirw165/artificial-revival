import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio

# ✅ Dynamically add the src/ folder to Python path
this_file = os.path.abspath(__file__)  # full path to full_project_animation.py
project_root = os.path.dirname(os.path.dirname(this_file))  # goes up to artificial-revival/
src_path = os.path.join(project_root, "src")
sys.path.append(src_path)

# ✅ Now you can safely import
from cryo_model import simulate_cell_viability
from revive_predictor import train_predictor


# Prepare output
frames_dir = "notebooks/final_frames"
os.makedirs(frames_dir, exist_ok=True)

def save_frame(fig, index):
    path = f"{frames_dir}/frame_{index:03d}.png"
    fig.savefig(path)
    plt.close(fig)

frame_index = 0

# -------- SCENE 1: Cell Viability --------
result = simulate_cell_viability()
fig, ax = plt.subplots(figsize=(8, 4), dpi=100)

ax.plot(result["time"], result["viability"], color='green', linewidth=2)
ax.set_title("Cell Viability Simulation")
ax.set_xlabel("Time (min)")
ax.set_ylabel("Viability (1=Alive, 0=Dead)")
fig.text(0.5, 0.01, "🧊 We simulate how a cell responds to freezing and osmotic stress.", ha='center', fontsize=12, color='blue')
save_frame(fig, frame_index); frame_index += 1

# -------- SCENE 2: AI Prediction --------
model = train_predictor()
labels = ['Dead', 'Alive']
counts = [model.predict([[1, 1, -1, -10, 2.5]])[0], 1 - model.predict([[1, 1, -1, -10, 2.5]])[0]]

fig, ax = plt.subplots(figsize=(8, 4), dpi=100)

ax.bar(labels, counts, color=['red', 'green'])
ax.set_title("AI Prediction: Will the Cell Survive?")
ax.set_ylim(0, 1.2)
fig.text(0.5, 0.01, "🤖 Our model learns to predict survival from cryo conditions.", ha='center', fontsize=12, color='blue')
save_frame(fig, frame_index); frame_index += 1

# -------- SCENE 3: EEG Simulation --------
frequencies = {"delta": 1.5, "theta": 5, "alpha": 10, "beta": 20, "gamma": 40}
duration = 5
sampling_rate = 256
t = np.linspace(0, duration, duration * sampling_rate)
np.random.seed(42)
signal = sum(np.sin(2 * np.pi * f * t + np.random.rand()) for f in frequencies.values())
signal += np.random.normal(0, 0.5, len(t))

fig, ax = plt.subplots(figsize=(8, 4), dpi=100)

ax.plot(t, signal, color='purple')
ax.set_title("Simulated Brain Signal (EEG-like)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Voltage")
fig.text(0.5, 0.01, "🧠 Neural activity before cryopreservation. Can identity be preserved?", ha='center', fontsize=12, color='blue')
save_frame(fig, frame_index); frame_index += 1

# -------- SCENE 4: Manifesto Quote --------
fig, ax = plt.subplots(figsize=(8, 4), dpi=100)

ax.axis('off')
fig.text(0.5, 0.6, "💭 'We do not aim to defy death —", ha='center', fontsize=14, style='italic')
fig.text(0.5, 0.5, "but to understand it, pause it, and revive with purpose.'", ha='center', fontsize=14, style='italic')
fig.text(0.5, 0.3, "— Mudassir Waheed", ha='center', fontsize=12, color='gray')
save_frame(fig, frame_index); frame_index += 1

# -------- Combine into GIF --------
images = [imageio.v2.imread(f"{frames_dir}/frame_{i:03d}.png") for i in range(frame_index)]
imageio.mimsave("notebooks/full_project_animation.gif", images, fps=1.2)

print("✅ Full project GIF created: notebooks/full_project_animation.gif")
