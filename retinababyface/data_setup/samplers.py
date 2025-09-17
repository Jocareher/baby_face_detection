# samplers.py
import math, random
from pathlib import Path
from collections import defaultdict, Counter, deque
from typing import Dict, List, Tuple, Any, Optional

import torch
from torch.utils.data import Sampler

# Mapea indices de orientación -> nombre (ajústalo a tu dataset si difiere)
class_names = {
    0: "left",
    1: "3_4_left",
    2: "frontal",
    3: "3_4_right",
    4: "right",
}


def _read_label_file(txt_path: Path) -> List[Tuple[int, float]]:
    """
    Lee todas las líneas de un .txt. Devuelve lista [(class_idx, child_prob), ...].
    Maneja archivo inexistente o vacío -> lista vacía.
    """
    if not txt_path.exists():
        return []
    lines = [ln.strip() for ln in txt_path.read_text().splitlines() if ln.strip()]
    pairs = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 2:
            continue
        try:
            cls = int(parts[0])
            child_prob = float(parts[1])
            pairs.append((cls, child_prob))
        except Exception:
            continue
    return pairs


def build_group_indices(
    dataset,
    child_thr: float = 0.5,
) -> Dict[str, List[int]]:
    """
    Agrupa indices del dataset en:
      child_left, child_3_4_left, child_frontal, child_3_4_right, child_right, adult_only, bg
    Reglas:
      - Si hay >=1 niño (child_prob>thr), elige la orientación de niño mayoritaria de esa imagen.
      - Si no hay niños y hay líneas -> adult_only
      - Si no hay .txt o .txt vacío -> bg
    """
    labels_dir = Path(dataset.root_dir) / dataset.split / "labels"
    groups = defaultdict(list)

    for i, stem in enumerate(dataset.file_list):
        txt = labels_dir / f"{stem}.txt"
        pairs = _read_label_file(txt)

        if not pairs:
            groups["bg"].append(i)
            continue

        # Niños por orientación
        child_orients = [
            cls for (cls, cp) in pairs if (cp > child_thr and 0 <= cls <= 4)
        ]
        if child_orients:
            cnt = Counter(child_orients)
            top_cls, _ = cnt.most_common(1)[0]
            groups[f"child_{class_names[top_cls]}"].append(i)
        else:
            # No hay niños -> adultos
            groups["adult_only"].append(i)

    # Asegura que existan todas las claves
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

    return dict(groups)


def scale_quota_for_batch_size(
    base_quota_32: Dict[str, int],
    batch_size: int,
    available: Dict[str, int],
    min_per_child: int = 1,
    priority_fill: Optional[List[str]] = None,
) -> Dict[str, int]:
    """
    Escala cuotas definidas para bs=32 a cualquier batch_size.
    - Redondea y asegura mínimo por clases de niño si hay disponibilidad.
    - Redistribuye sobrantes o déficits en orden de prioridad.
    """
    if priority_fill is None:
        # Primero minoritarias de niño, luego 3/4, luego frontal, luego adulto/bg
        priority_fill = [
            "child_left",
            "child_right",
            "child_3_4_left",
            "child_3_4_right",
            "child_frontal",
            "adult_only",
            "bg",
        ]

    # 1) Escalado lineal
    quota = {}
    for k, v in base_quota_32.items():
        q = int(round(v * batch_size / 32.0))
        quota[k] = q

    # 2) Mínimos para niños si el grupo existe
    for k in [
        "child_left",
        "child_right",
        "child_3_4_left",
        "child_3_4_right",
        "child_frontal",
    ]:
        if available.get(k, 0) > 0:
            quota[k] = max(quota.get(k, 0), min_per_child)

    # 3) Ajuste para que la suma == batch_size
    total = sum(quota.values())

    def inc(k):
        quota[k] = quota.get(k, 0) + 1

    def dec(k):
        quota[k] = max(0, quota.get(k, 0) - 1)

    if total < batch_size:
        # rellenar
        deficit = batch_size - total
        j = 0
        while deficit > 0:
            k = priority_fill[j % len(priority_fill)]
            # sólo sumamos si hay material en ese grupo
            if available.get(k, 0) > 0:
                inc(k)
                deficit -= 1
            j += 1
    elif total > batch_size:
        # recortar empezando por bg/adult/frontal
        order = list(reversed(priority_fill))  # quita primero bg/adult/frontal
        excess = total - batch_size
        j = 0
        while excess > 0:
            k = order[j % len(order)]
            if quota.get(k, 0) > 0:
                dec(k)
                excess -= 1
            j += 1

    # Si algún grupo no existe, pon 0
    for k, n in available.items():
        if n == 0:
            quota[k] = 0

    return quota


class StratifiedBatchSampler(Sampler[List[int]]):
    """
    Devuelve lotes estratificados respetando cuota por grupo.
    - No repite indices dentro del mismo batch.
    - Con replacement=True puede repetir a lo largo de la época si el grupo se agota.
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
        self.groups = {k: list(v) for k, v in groups.items()}
        self.batch_quota = dict(batch_quota)
        self.n_batches = n_batches
        self.replacement = replacement
        self.drop_last = drop_last
        self.rng = random.Random(seed)

        # Crea colas por grupo
        self.pools = {}
        for k, idxs in self.groups.items():
            self.rng.shuffle(idxs)
            self.pools[k] = deque(idxs)

        # Pre-chequeo: suma de cuotas
        self.batch_size = sum(self.batch_quota.values())
        assert self.batch_size > 0, "batch_quota suma 0."

    def __len__(self):
        return self.n_batches if self.drop_last else self.n_batches

    def _draw_from_group(self, k: str, q: int) -> List[int]:
        out = []
        pool = self.pools[k]
        for _ in range(q):
            if pool:
                out.append(pool.popleft())
            else:
                if not self.replacement or len(self.groups[k]) == 0:
                    # Sin material: salta (o podrías robar de otros grupos)
                    continue
                # Re-arma el pool con shuffle y saca uno
                refill = list(self.groups[k])
                self.rng.shuffle(refill)
                pool.extend(refill)
                out.append(pool.popleft())
        # Evita duplicados accidentales si q>len(grupo) y replacement=False (raro)
        return list(dict.fromkeys(out))

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            for k, q in self.batch_quota.items():
                if q <= 0:
                    continue
                batch.extend(self._draw_from_group(k, q))

            # Si por falta de material el batch quedó corto, rellena al vuelo
            if len(batch) < self.batch_size:
                # pick desde todos los grupos con material
                flat = [i for v in self.groups.values() for i in v]
                if flat:
                    need = self.batch_size - len(batch)
                    self.rng.shuffle(flat)
                    add = (
                        flat[:need]
                        if not self.replacement
                        else [self.rng.choice(flat) for _ in range(need)]
                    )
                    # evita duplicados dentro del mismo batch
                    seen = set(batch)
                    for a in add:
                        if a not in seen:
                            batch.append(a)
                            seen.add(a)
                        if len(batch) == self.batch_size:
                            break

            yield batch


def make_stratified_batch_sampler(
    dataset,
    batch_size: int,
    seed: int = 42,
    replacement: bool = True,
    drop_last: bool = True,
):
    groups = build_group_indices(dataset, child_thr=0.5)

    # Disponibilidad por grupo
    available = {k: len(v) for k, v in groups.items()}

    # Cuota base pensada para bs=32 (ajústala si quieres otra priorización)
    base_quota_32 = {
        "child_left": 3,
        "child_3_4_left": 4,
        "child_frontal": 6,
        "child_3_4_right": 4,
        "child_right": 3,
        "adult_only": 6,
        "bg": 6,
    }  # suma = 32

    batch_quota = scale_quota_for_batch_size(
        base_quota_32=base_quota_32,
        batch_size=batch_size,
        available=available,
        min_per_child=1 if batch_size <= 16 else 2,  # mínimos por orientación
    )

    # nº de batches por época
    n_batches = (
        len(dataset) // batch_size
        if drop_last
        else math.ceil(len(dataset) / batch_size)
    )

    sampler = StratifiedBatchSampler(
        groups=groups,
        batch_quota=batch_quota,
        n_batches=n_batches,
        seed=seed,
        replacement=replacement,
        drop_last=drop_last,
    )

    return sampler, {"groups": available, "quota": batch_quota}
