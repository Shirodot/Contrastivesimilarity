# Assignment 1 — Contrastive Learning


> Given the distance loss for **negative samples** in contrastive learning:
>
> ```
> L = max(0, C - ||D(x1) - D(x2)||_2)
> ```

Compare **any number of pictures** using contrastive learning loss to measure similarity between images.

---

## Quick Start (Automatic Mode)

### 1. Add images to the `images/` folder

```bash
# The images/ folder is created automatically when you run the program
# Just copy your images there:
images/
├── Cat.jpg
├── Dog.jpg
├── Husky.jpg
└── Bird.png
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.webp`

### 2. Compile and run

```bash
javac ContrastiveSimilarity.java
java  ContrastiveSimilarity
```

**That's it!** The program will:
- ✅ Auto-create the `images/` folder if needed
- ✅ Auto-assign **labels based on filenames** (e.g., `Cat.jpg` → `Cat`)
- ✅ Auto-generate **feature values** for each image
- ✅ Compare **all pairs** automatically

**No `.txt` files required!**

---

## Optional: Custom Features with `.txt` Files

To **override auto-generated features** with your own values, create a `.txt` file for any image:

```
images/
├── cat.jpg
├── cat.txt          ← optional custom features
├── labrador.jpg
└── labrador.txt
```

**Format of `.txt` file:**
```
label=cat
fur_texture=0.3
ear_shape=0.8
body_size=0.2
snout_length=0.1
```

### Scoring Guide (0.0 = low → 1.0 = high)

| Feature | 0.0 (low) | 1.0 (high) |
|---------|-----------|------------|
| `fur_texture` | smooth / short fur | thick / fluffy fur |
| `ear_shape` | round / floppy ears | tall / pointed ears |
| `body_size` | tiny / small | large / big |
| `snout_length` | flat face | long snout |

The `label` can be **anything** — `cat`, `dog`, `bird`, `rabbit`, `fish`, etc.

---

## Example Output

### Input:
3 images: `Cat.jpg`, `Dog.jpg`, `Husky.jpg`

### Output:
```
[Loading] Scanning images/

-----------------------------------------------------------------
  Cat.jpg                   -> OK  label=Cat         (156 KB)  [auto-generated]
  Dog.jpg                   -> OK  label=Dog         (203 KB)  [auto-generated]
  Husky.jpg                 -> OK  label=Husky       (189 KB)  [auto-generated]
-----------------------------------------------------------------

  Loaded 3 subject(s).

[Summary] All Pairs

  Cat / Dog                36.45  52.4%  0.4823  Negative
  Cat / Husky              41.23  48.2%  0.5179  Negative
  Dog / Husky              18.92  88.3%  0.1169  Negative

[Stats]

  Most similar pair  : Dog / Husky  (88.3%)
  Least similar pair : Cat / Husky  (48.2%)
```

**Key insight:** Because `Cat`, `Dog`, and `Husky` are different animal types, they naturally show **lower similarity** and **higher loss** values (they need to be pushed apart in contrastive learning).

---

## How Contrastive Learning Works

### The Loss Formula

```
L = max(0, C - ||D(x1) - D(x2)||_2)
```

Where:
- `D(x)` = the feature embedding (4 features: fur_texture, ear_shape, body_size, snout_length)
- `||D(x1) - D(x2)||_2` = L2 (Euclidean) distance between two embeddings
- `C` = margin constant (default: 1.0)

### Two Types of Pairs

| Pair Type | Examples | Goal | Expected |
|-----------|----------|------|----------|
| **Positive** (same class) | cat + cat | Pull close | Low L2, Loss ≈ 0 |
| **Negative** (different class) | cat + dog | Push apart | High L2, Loss ≈ 0 |

### Loss Interpretation

| Condition | Loss | Meaning |
|-----------|------|---------|
| L2 distance ≥ C | L = 0 | ✓ Pairs are far apart — no penalty |
| L2 distance < C | L > 0 | ✗ Pairs too close — model is penalised |

---

## File Structure

```
.
├── ContrastiveSimilarity.java   Main program
├── README.md                    This file
├── gitignore                    Git ignore rules
└── images/                      Images folder (auto-created)
    ├── Cat.jpg
    ├── Dog.jpg
    ├── Husky.jpg
    └── (optional) Dog.txt       Custom features override
```

---

## Tips

- **Use descriptive filenames** — The filename becomes the label
- **Custom precision** — Only create `.txt` files for images where you want precise control
- **Same class images** — Name them similarly: `Cat_A.jpg`, `Cat_B.jpg`

---

## License

For educational purposes only.
