import os
import torch
import matplotlib.pyplot as plt

# -----------------------------
# Editable baseline values
# -----------------------------
# You can tune these 4 numbers directly.
TRAIN_WMMSE_BASELINE = 20.902
TRAIN_LMMSE_BASELINE = 16.627
TEST_WMMSE_BASELINE = 20.902
TEST_LMMSE_BASELINE = 16.627


def _to_1d_tensor(x):
	if torch.is_tensor(x):
		return x.detach().cpu().flatten().float()
	return torch.tensor(x, dtype=torch.float32).flatten()


def load_rate_pt(pt_path, preferred_key):
	"""Load epochs and rate from .pt.

	Supports formats like:
	- {'epochs': ..., 'train_rate' / 'test_rate': ...}
	- {'rate': ...}
	- raw tensor/list
	"""
	obj = torch.load(pt_path, map_location="cpu")

	if isinstance(obj, dict):
		# rate key
		if preferred_key in obj:
			rate = _to_1d_tensor(obj[preferred_key])
		elif "rate" in obj:
			rate = _to_1d_tensor(obj["rate"])
		else:
			# fallback: first tensor/list-like value
			picked = None
			for v in obj.values():
				if torch.is_tensor(v) or isinstance(v, (list, tuple)):
					picked = v
					break
			if picked is None:
				raise ValueError(f"Cannot find rate series in {pt_path}")
			rate = _to_1d_tensor(picked)

		# epoch key
		if "epochs" in obj:
			epochs = _to_1d_tensor(obj["epochs"])
			if epochs.numel() != rate.numel():
				epochs = torch.arange(1, rate.numel() + 1, dtype=torch.float32)
		else:
			epochs = torch.arange(1, rate.numel() + 1, dtype=torch.float32)
	else:
		rate = _to_1d_tensor(obj)
		epochs = torch.arange(1, rate.numel() + 1, dtype=torch.float32)

	return epochs.numpy(), rate.numpy()


def plot_single_curve(epochs, rates, title, color, wmmse_baseline, lmmse_baseline):
	plt.figure(figsize=(8, 5))
	plt.plot(epochs, rates, linewidth=1.8, color=color)
	plt.axhline(y=wmmse_baseline, color="#2ca02c", linestyle="--", linewidth=1.6,
				label=f"WMMSE baseline = {wmmse_baseline:.3f}")
	plt.axhline(y=lmmse_baseline, color="#9467bd", linestyle=":", linewidth=1.8,
				label=f"LMMSE baseline = {lmmse_baseline:.3f}")
	plt.xlabel("Epoch")
	plt.ylabel("Sum rate")
	plt.title(title)
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.show()


def main():
	root = os.path.dirname(os.path.abspath(__file__))
	# train_pt = os.path.join(root, "training_rate.pt")
	train_pt = os.path.join(root, "a1a_training_rate.pt")
	test_pt = os.path.join(root, "testing_rate.pt")

	if not os.path.exists(train_pt):
		raise FileNotFoundError(f"Not found: {train_pt}")
	if not os.path.exists(test_pt):
		raise FileNotFoundError(f"Not found: {test_pt}")

	train_epochs, train_rate = load_rate_pt(train_pt, preferred_key="train_rate")
	test_epochs, test_rate = load_rate_pt(test_pt, preferred_key="test_rate")

	train_rate = train_rate * 1.0
	test_rate = test_rate * 1.0

	plot_single_curve(
		train_epochs,
		train_rate,
		"Train Sum-Rate vs Epoch",
		color="#1f77b4",
		wmmse_baseline=TRAIN_WMMSE_BASELINE,
		lmmse_baseline=TRAIN_LMMSE_BASELINE,
	)
	plot_single_curve(
		test_epochs,
		test_rate,
		"Test Sum-Rate vs Epoch",
		color="#d62728",
		wmmse_baseline=TEST_WMMSE_BASELINE,
		lmmse_baseline=TEST_LMMSE_BASELINE,
	)


if __name__ == "__main__":
	main()

