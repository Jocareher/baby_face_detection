# samplers.py
import math, random
from pathlib import Path
from collections import defaultdict, Counter, deque
from typing import Dict, List, Tuple, Any, Optional

import torch
from torch.utils.data import Sampler, WeightedRandomSampler

# Map from class index to orientation name
class_names = {
    0: "left",
    1: "3_4_left",
    2: "frontal",
    3: "3_4_right",
    4: "right",
}


def read_label_file(txt_path: Path) -> List[Tuple[int, float]]:
    """
    Reads a label file and extracts class indices and child probabilities.

    This function processes a .txt file where each line contains a class index
    followed by a child probability. It handles cases where the file does not
    exist or is empty by returning an empty list.

    Args:
        txt_path (Path): The path to the label file.

    Returns:
        List[Tuple[int, float]]: A list of tuples, each containing a class index
        and its corresponding child probability. If the file is missing or empty,
        returns an empty list.
    """
    if not txt_path.exists():
        return []  # Return an empty list if the file does not exist.

    # Read lines from the file, stripping whitespace and ignoring empty lines.
    lines = [ln.strip() for ln in txt_path.read_text().splitlines() if ln.strip()]
    pairs = []  # Initialize a list to store (class_idx, child_prob) pairs.

    for ln in lines:
        parts = ln.split()  # Split the line into parts.
        if len(parts) < 2:
            continue  # Skip lines that do not have at least two parts.
        try:
            cls = int(parts[0])  # Parse the class index.
            child_prob = float(parts[1])  # Parse the child probability.
            pairs.append((cls, child_prob))  # Append the tuple to the list.
        except ValueError:
            continue  # Skip lines with invalid data.

    return pairs  # Return the list of (class_idx, child_prob) tuples.


def build_group_indices(
    dataset,
    child_thr: float = 0.5,
) -> Dict[str, List[int]]:
    """
    Groups dataset indices based on the presence of children and their orientations.

    This function categorizes indices into the following groups:
      - child_left
      - child_3_4_left
      - child_frontal
      - child_3_4_right
      - child_right
      - adult_only
      - bg (background)

    Rules for grouping:
      - If there is at least one child (child_prob > child_thr), the predominant orientation of the child in the image is selected.
      - If no children are present but there are labels, the image is categorized as adult_only.
      - If the label file does not exist or is empty, the image is categorized as bg.

    Args:
        dataset: The dataset containing images and their associated label files.
        child_thr (float): The threshold for considering a child present based on child probability.

    Returns:
        Dict[str, List[int]]: A dictionary where keys are group names and values are lists of indices corresponding to those groups.
    """
    labels_dir = Path(dataset.root_dir) / dataset.split / "labels"
    groups = defaultdict(list)

    for i, stem in enumerate(dataset.file_list):
        txt = labels_dir / f"{stem}.txt"
        pairs = read_label_file(txt)

        if not pairs:
            groups["bg"].append(i)  # No labels found, categorize as background.
            continue

        # Extract child orientations based on probabilities
        child_orients = [
            cls for (cls, cp) in pairs if (cp > child_thr and 0 <= cls <= 4)
        ]
        if child_orients:
            cnt = Counter(child_orients)  # Count occurrences of each orientation.
            top_cls, _ = cnt.most_common(1)[0]  # Get the most common orientation.
            groups[f"child_{class_names[top_cls]}"].append(
                i
            )  # Categorize by predominant orientation.
        else:
            # No children detected, categorize as adult_only.
            groups["adult_only"].append(i)

    # Ensure all group keys exist in the dictionary
    for k in [
        "child_left",
        "child_3_4_left",
        "child_frontal",
        "child_3_4_right",
        "child_right",
        "adult_only",
        "bg",
    ]:
        groups.setdefault(k, [])

    return dict(groups)  # Return the grouped indices as a standard dictionary.


def scale_quota_for_batch_size(
    base_quota_32: Dict[str, int],
    batch_size: int,
    available: Dict[str, int],
    min_per_child: int = 1,
    priority_fill: Optional[List[str]] = None,
) -> Dict[str, int]:
    """
    Scales the defined quotas for a batch size of 32 to any specified batch size.

    This function adjusts the quotas for different groups based on the desired batch size,
    ensuring that minimum quotas for child classes are respected if available. It redistributes
    any surplus or deficit in quotas according to a specified priority order.

    Args:
        base_quota_32 (Dict[str, int]): A dictionary defining the base quotas for a batch size of 32.
        batch_size (int): The desired batch size to scale the quotas for.
        available (Dict[str, int]): A dictionary indicating the number of available samples for each group.
        min_per_child (int, optional): The minimum number of samples to allocate for child classes if available. Defaults to 1.
        priority_fill (Optional[List[str]], optional): A list defining the order of priority for filling quotas.
            If None, defaults to a predefined order.

    Returns:
        Dict[str, int]: A dictionary with the scaled quotas for each group, ensuring the total equals the specified batch size.
    """
    if priority_fill is None:
        # Define the priority order for filling quotas
        priority_fill = [
            "child_left",
            "child_right",
            "child_3_4_left",
            "child_3_4_right",
            "child_frontal",
            "adult_only",
            "bg",
        ]

    # 1) Linear scaling of quotas based on the batch size
    quota = {}
    for k, v in base_quota_32.items():
        q = int(round(v * batch_size / 32.0))
        quota[k] = q

    # 2) Ensure minimum quotas for child classes if available
    for k in [
        "child_left",
        "child_right",
        "child_3_4_left",
        "child_3_4_right",
        "child_frontal",
    ]:
        if available.get(k, 0) > 0:
            quota[k] = max(quota.get(k, 0), min_per_child)

    # 3) Adjust quotas to ensure the total equals the batch size
    total = sum(quota.values())

    def inc(k):
        quota[k] = quota.get(k, 0) + 1

    def dec(k):
        quota[k] = max(0, quota.get(k, 0) - 1)

    if total < batch_size:
        # Fill the deficit
        deficit = batch_size - total
        j = 0
        while deficit > 0:
            k = priority_fill[j % len(priority_fill)]
            if available.get(k, 0) > 0:
                inc(k)
                deficit -= 1
            j += 1
    elif total > batch_size:
        # Trim excess starting from less prioritized groups
        order = list(reversed(priority_fill))  # Remove from bg/adult/frontal first
        excess = total - batch_size
        j = 0
        while excess > 0:
            k = order[j % len(order)]
            if quota.get(k, 0) > 0:
                dec(k)
                excess -= 1
            j += 1

    # Set quota to 0 for any group that has no available samples
    for k, n in available.items():
        if n == 0:
            quota[k] = 0

    return quota


class StratifiedBatchSampler(Sampler[List[int]]):
    """
    A sampler that returns stratified batches respecting the quota for each group.

    This sampler ensures that:
    - No indices are repeated within the same batch.
    - If replacement is enabled, indices can be repeated across epochs if a group is exhausted.

    Args:
        groups (Dict[str, List[int]]): A dictionary where keys are group names and values are lists of indices for each group.
        batch_quota (Dict[str, int]): A dictionary defining the number of samples to draw from each group for a batch.
        n_batches (int): The total number of batches to generate.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        replacement (bool, optional): If True, allows indices to be reused across batches. Defaults to True.
        drop_last (bool, optional): If True, drops the last batch if it is smaller than the specified batch size. Defaults to True.
    """

    def __init__(
        self,
        groups: Dict[str, List[int]],
        batch_quota: Dict[str, int],
        n_batches: int,
        seed: int = 42,
        replacement: bool = True,
        drop_last: bool = True,
    ):
        self.groups = {k: list(v) for k, v in groups.items()}  # Store groups as lists
        self.batch_quota = dict(batch_quota)  # Store batch quotas
        self.n_batches = n_batches  # Total number of batches
        self.replacement = replacement  # Replacement flag
        self.drop_last = drop_last  # Drop last batch flag
        self.rng = random.Random(seed)  # Random number generator

        # Create queues for each group
        self.pools = {}
        for k, idxs in self.groups.items():
            self.rng.shuffle(idxs)  # Shuffle indices for randomness
            self.pools[k] = deque(idxs)  # Use deque for efficient pops

        # Pre-check: sum of quotas must be greater than zero
        self.batch_size = sum(self.batch_quota.values())
        assert self.batch_size > 0, "batch_quota sum must be greater than 0."

    def __len__(self):
        return self.n_batches if self.drop_last else self.n_batches

    def draw_from_group(self, k: str, q: int) -> List[int]:
        """
        Draws samples from a specified group.

        Args:
            k (str): The group key from which to draw samples.
            q (int): The number of samples to draw.

        Returns:
            List[int]: A list of drawn indices from the specified group.
        """
        out = []
        pool = self.pools[k]  # Get the pool for the group
        for _ in range(q):
            if pool:
                out.append(pool.popleft())  # Draw from the pool
            else:
                if not self.replacement or len(self.groups[k]) == 0:
                    # If no material and no replacement, skip
                    continue
                # Refill the pool and shuffle if exhausted
                refill = list(self.groups[k])
                self.rng.shuffle(refill)
                pool.extend(refill)
                out.append(pool.popleft())  # Draw from the refilled pool
        # Avoid accidental duplicates if q > len(group) and replacement is False
        return list(dict.fromkeys(out))

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            for k, q in self.batch_quota.items():
                if q <= 0:
                    continue
                batch.extend(self.draw_from_group(k, q))  # Draw samples for the batch

            # If the batch is short due to lack of material, fill on the fly
            if len(batch) < self.batch_size:
                # Pick from all groups with available material
                flat = [i for v in self.groups.values() for i in v]
                if flat:
                    need = self.batch_size - len(batch)
                    self.rng.shuffle(flat)
                    add = (
                        flat[:need]
                        if not self.replacement
                        else [self.rng.choice(flat) for _ in range(need)]
                    )
                    # Avoid duplicates within the same batch
                    seen = set(batch)
                    for a in add:
                        if a not in seen:
                            batch.append(a)
                            seen.add(a)
                        if len(batch) == self.batch_size:
                            break

            yield batch  # Yield the constructed batch


def make_stratified_batch_sampler(
    dataset,
    batch_size: int,
    seed: int = 42,
    replacement: bool = True,
    drop_last: bool = True,
):
    """
    Creates a stratified batch sampler for the given dataset.

    This function builds a sampler that generates batches of data while maintaining
    the specified distribution of classes. It ensures that each batch contains a
    balanced representation of different groups based on the defined quotas.

    Args:
        dataset: The dataset containing images and their associated label files.
        batch_size (int): The desired size of each batch.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        replacement (bool, optional): If True, allows indices to be reused across batches. Defaults to True.
        drop_last (bool, optional): If True, drops the last batch if it is smaller than the specified batch size. Defaults to True.

    Returns:
        Tuple[StratifiedBatchSampler, Dict[str, Dict[str, int]]]: A tuple containing the stratified batch sampler
        and a dictionary with the available groups and their corresponding quotas.
    """
    # Build group indices based on the dataset
    groups = build_group_indices(dataset, child_thr=0.5)

    # Count the number of available samples in each group
    available = {k: len(v) for k, v in groups.items()}

    # Define base quotas for a batch size of 32
    base_quota_32 = {
        "child_left": 3,
        "child_3_4_left": 4,
        "child_frontal": 6,
        "child_3_4_right": 4,
        "child_right": 3,
        "adult_only": 6,
        "bg": 6,
    }  # Total sum = 32

    # Scale the quotas based on the desired batch size
    batch_quota = scale_quota_for_batch_size(
        base_quota_32=base_quota_32,
        batch_size=batch_size,
        available=available,
        min_per_child=1
        if batch_size <= 16
        else 2,  # Minimum samples per child orientation
    )

    # Calculate the number of batches per epoch
    n_batches = (
        len(dataset) // batch_size
        if drop_last
        else math.ceil(len(dataset) / batch_size)
    )

    # Create the stratified batch sampler
    sampler = StratifiedBatchSampler(
        groups=groups,
        batch_quota=batch_quota,
        n_batches=n_batches,
        seed=seed,
        replacement=replacement,
        drop_last=drop_last,
    )

    return sampler, {"groups": available, "quota": batch_quota}


labels_maps = {
    -2: "BG",
    -1: "Adult",
    0: "Leftside",
    1: "3/4 Leftside",
    2: "Frontal",
    3: "3/4 Rightside",
    4: "Rightside",
}


def dominant_group_for_label_file(txt_path: Path) -> int:
    """
    Determines the dominant group for a given label file.

    This function analyzes a label file to identify the predominant group based on the
    class indices present in the file. The rules for determining the dominant group are as follows:
      - If the file does not exist or is empty, it returns BG (-2).
      - It counts all lines in the file; lines with class_idx == -1 are counted as Adult.
      - Class indices in the range [0..4] are counted towards their respective orientations.
      - The group with the highest count is considered the dominant group. In case of a tie,
        non-BG groups are prioritized if they exist.

    Args:
        txt_path (Path): The path to the label file to be analyzed.

    Returns:
        int: The dominant group index. Possible values include:
            - -2: Background (BG)
            - -1: Adult
            - 0 to 4: Corresponding to different orientations (left, 3/4 left, frontal, 3/4 right, right).
    """
    if (not txt_path.exists()) or txt_path.stat().st_size == 0:
        return -2  # BG

    counts = Counter()  # Initialize a counter to track occurrences of each class
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()  # Remove leading/trailing whitespace
            if not line:
                continue  # Skip empty lines
            parts = line.split()  # Split the line into parts
            try:
                cls = int(float(parts[0]))  # Parse the class index
            except Exception:
                continue  # Skip lines with invalid data
            if cls == -1:
                counts[-1] += 1  # Count as Adult
            elif 0 <= cls <= 4:
                counts[cls] += 1  # Count towards the respective orientation

    if not counts:
        return -2  # If no interpretable data, treat as BG

    # Determine the dominant group based on the counts
    group, _ = counts.most_common(1)[0]  # Get the most common group
    return group


def scan_dataset_groups(dataset) -> Tuple[List[int], Dict[int, int]]:
    """
    Scans the dataset to determine the group associated with each image and
    calculates the frequency of each group.

    This function reads the label files for each image in the dataset and
    identifies the dominant group for each image using the
    `dominant_group_for_label_file` function. It returns a list of group
    indices corresponding to each image and a frequency count of how many
    images belong to each group.

    Args:
        dataset: The dataset containing images and their associated label files.

    Returns:
        Tuple[List[int], Dict[int, int]]: A tuple containing:
            - groups_per_image: A list of group indices (one for each image).
            - freqs: A dictionary (Counter) with the frequency of each group.
    """
    labels_dir = Path(dataset.root_dir) / dataset.split / "labels"
    label_files = [labels_dir / f"{stem}.txt" for stem in dataset.file_list]

    groups = []
    for txt in label_files:
        g = dominant_group_for_label_file(
            txt
        )  # Determine the dominant group for the label file
        groups.append(g)  # Append the group index to the list

    freqs = Counter(groups)  # Count the frequency of each group
    return groups, freqs  # Return the list of groups and their frequencies


def make_weighted_sampler(dataset, smooth: float = 0.0, seed: int = 42):
    """
    Creates a weighted sampler for the dataset, assigning weights to each image
    inversely proportional to the frequency of its dominant group. This helps
    balance the sampling across different groups, especially in cases where
    some groups may be underrepresented.

    Args:
        dataset: The dataset containing images and their associated label files.
        smooth (float, optional): A smoothing factor to apply to the frequencies.
            If greater than 0, it helps to mitigate the impact of very low
            frequencies by adding this value to the counts. Defaults to 0.0.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[WeightedRandomSampler, Dict[str, Any]]: A tuple containing:
            - sampler: A WeightedRandomSampler instance for drawing samples.
            - info: A dictionary with the frequency of each group and the total
              number of images in the dataset.
    """
    # Scan the dataset to get the dominant group for each image and their frequencies
    groups, freqs = scan_dataset_groups(dataset)

    # Calculate inverse weights with optional smoothing: 1/(frequency + smooth)
    weights = []
    for g in groups:
        f = float(freqs[g])  # Get the frequency of the dominant group
        w = 1.0 / (f + smooth) if f > 0 else 0.0  # Calculate weight
        weights.append(w)  # Append weight to the list

    # Convert weights to a tensor
    weights = torch.tensor(weights, dtype=torch.float)

    # Create a WeightedRandomSampler with the calculated weights
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(weights),  # Number of samples equals the dataset size
        replacement=True,  # Allow replacement
        generator=torch.Generator().manual_seed(seed),  # Set random seed
    )

    # Prepare information about the frequencies and number of images
    info = {
        "freqs": {
            labels_maps[k]: int(v) for k, v in freqs.items()
        },  # Map frequencies to labels
        "num_images": len(groups),  # Total number of images in the dataset
    }
    return sampler, info  # Return the sampler and info
