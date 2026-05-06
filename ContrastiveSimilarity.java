import java.io.*;
import java.util.*;

/**
 * =============================================================================
 * Assignment 1 - Contrastive Learning
 * Course   : Algorithms
 * Lecturer : Xiang-Rui Huang, NPUST / NYCU
 *
 * Formula  : L = max(0, C - ||D(x1) - D(x2)||_2)
 *
 * HOW TO USE:
 *   1. Drop ANY number of images into the /images folder
 *   2. For each image, create a matching .txt file in /images with the same
 *      name, containing the label and 4 feature scores. Example:
 *
 *        File: images/my_cat.jpg
 *        Meta: images/my_cat.txt  ← content below
 *              label=cat
 *              fur_texture=0.9
 *              ear_shape=0.3
 *              body_size=0.2
 *              snout_length=0.1
 *
 *   3. Compile and run:
 *        javac ContrastiveSimilarity.java
 *        java  ContrastiveSimilarity
 *
 * Scoring guide (0.0 = low, 1.0 = high):
 *   fur_texture  : 0.0 smooth/short   →  1.0 thick/fluffy
 *   ear_shape    : 0.0 round/floppy   →  1.0 tall/pointed
 *   body_size    : 0.0 tiny/small     →  1.0 large/big
 *   snout_length : 0.0 flat/no snout  →  1.0 long snout
 * =============================================================================
 */
public class ContrastiveSimilarity {

    static final double MARGIN_C  = 1.0;
    static final String IMAGE_DIR = "images";
    static final String[] FEATURE_NAMES = {
        "fur_texture", "ear_shape", "body_size", "snout_length"
    };

    // =========================================================================
    // Animal / Subject record
    // =========================================================================
    static class Subject {
        String   name;       // derived from filename
        String   label;      // e.g. "cat", "dog", "bird" — anything you set
        String   filename;
        double[] embedding;  // D(x): feature vector

        Subject(String name, String label, String filename, double[] embedding) {
            this.name      = name;
            this.label     = label;
            this.filename  = filename;
            this.embedding = embedding;
        }
    }

    // =========================================================================
    // Step 1 - Scan /images folder and load all subjects from .txt metadata
    // =========================================================================
    static List<Subject> loadSubjects() {
        List<Subject> subjects = new ArrayList<>();
        File dir = new File(IMAGE_DIR);

        // Automatically create the folder if it doesn't exist
        if (!dir.exists()) {
            boolean created = dir.mkdirs();
            if (created) {
                System.out.println("  [INFO] Created folder: " + IMAGE_DIR + "/");
            } else {
                System.out.println("  [ERROR] Failed to create folder: " + IMAGE_DIR + "/");
                return subjects;
            }
        }

        if (!dir.isDirectory()) {
            System.out.println("  [ERROR] Path exists but is not a folder: " + IMAGE_DIR + "/");
            return subjects;
        }

        // accepted image extensions
        Set<String> imgExts = new HashSet<>(Arrays.asList("jpg","jpeg","png","gif","bmp","webp"));

        File[] files = dir.listFiles();
        if (files == null || files.length == 0) {
            System.out.println("  [!] images/ folder is empty.");
            return subjects;
        }
        Arrays.sort(files, Comparator.comparing(File::getName));

        System.out.println("\n[Loading] Scanning " + IMAGE_DIR + "/\n");
        line();

        for (File f : files) {
            String fname = f.getName();
            String ext   = extension(fname).toLowerCase();
            if (!imgExts.contains(ext)) continue; // skip non-images

            // look for matching .txt metadata file
            String baseName = fname.substring(0, fname.lastIndexOf('.'));
            File   metaFile = new File(IMAGE_DIR + File.separator + baseName + ".txt");

            String label = baseName;  // Use filename as label by default
            double[] emb = new double[FEATURE_NAMES.length];

            // If .txt file exists, parse it; otherwise use default values
            if (metaFile.exists()) {
                try {
                    Map<String, String> props = parseProperties(metaFile);
                    label = props.getOrDefault("label", baseName);
                    for (int i = 0; i < FEATURE_NAMES.length; i++) {
                        emb[i] = Double.parseDouble(
                            props.getOrDefault(FEATURE_NAMES[i], "0.5")
                        );
                    }
                } catch (Exception e) {
                    System.out.printf("  %-25s -> ERROR reading %s.txt: %s%n",
                            fname, baseName, e.getMessage());
                    continue;
                }
            } else {
                // Auto-generate semantic features based on category
                emb = generateSemanticFeatures(baseName);
            }

            long kb = f.length() / 1024;
            String source = metaFile.exists() ? "metadata" : "auto-generated";
            System.out.printf("  %-25s -> OK  label=%-12s  (%d KB)  [%s]%n",
                    fname, label, kb, source);
            subjects.add(new Subject(baseName, label, fname, emb));
        }
        line();

        if (subjects.isEmpty()) {
            System.out.println("\n  [!] No valid subjects loaded.");
            System.out.println("      Make sure each image has a matching .txt file.\n");
        } else {
            System.out.println("\n  Loaded " + subjects.size() + " subject(s).\n");
        }
        return subjects;
    }

    // =========================================================================
    // Step 2 - Distance Functions
    // =========================================================================

    /** L1 (Manhattan): ||x1 - x2||_1 */
    static double l1Distance(double[] x1, double[] x2) {
        double sum = 0.0;
        for (int i = 0; i < x1.length; i++) sum += Math.abs(x1[i] - x2[i]);
        return sum;
    }

    /** L2 (Euclidean): ||x1 - x2||_2 */
    static double l2Distance(double[] x1, double[] x2) {
        double sum = 0.0;
        for (int i = 0; i < x1.length; i++) {
            double d = x1[i] - x2[i];
            sum += d * d;
        }
        return Math.sqrt(sum);
    }

    // =========================================================================
    // Step 2b - Semantic Feature Generation (Auto-categorization)
    // =========================================================================

    /**
     * Auto-generates semantic features based on the label/filename.
     * Same-category items get similar features, different categories are distinct.
     * Features: [fur_texture, ear_shape, body_size, snout_length]
     */
    static double[] generateSemanticFeatures(String label) {
        String lower = label.toLowerCase();
        double[] features = new double[4];

        // ANIMALS
        if (lower.contains("cat")) {
            features = new double[]{0.7, 0.8, 0.25, 0.15}; // fluffy, pointy ears, small, short snout
        } else if (lower.contains("dog") || lower.contains("labrador") || lower.contains("poodle")) {
            features = new double[]{0.85, 0.55, 0.60, 0.65}; // fluffy, floppy ears, medium, medium snout
        } else if (lower.contains("husky") || lower.contains("shepherd") || lower.contains("wolf")) {
            features = new double[]{0.90, 0.75, 0.70, 0.70}; // very fluffy, pointy ears, large, long snout
        } else if (lower.contains("bird") || lower.contains("parrot") || lower.contains("eagle")) {
            features = new double[]{0.45, 0.65, 0.15, 0.35}; // feathers>fur, pointy ears, tiny, medium snout
        } else if (lower.contains("fish") || lower.contains("shark")) {
            features = new double[]{0.20, 0.10, 0.40, 0.25}; // scales, no ears, medium, short snout
        }
        // VEHICLES
        else if (lower.contains("747") || lower.contains("plane") || lower.contains("aircraft") || lower.contains("jet")) {
            features = new double[]{0.05, 0.05, 0.95, 0.05}; // no fur, no ears, huge, no snout
        } else if (lower.contains("car") || lower.contains("vehicle") || lower.contains("sedan") || lower.contains("truck")) {
            features = new double[]{0.10, 0.08, 0.55, 0.08}; // no fur, no ears, medium, no snout
        } else if (lower.contains("bike") || lower.contains("motorcycle")) {
            features = new double[]{0.08, 0.10, 0.35, 0.10}; // no fur, no ears, small, no snout
        } else if (lower.contains("boat") || lower.contains("ship")) {
            features = new double[]{0.12, 0.10, 0.80, 0.15}; // no fur, no ears, large, short snout
        }
        // FALLBACK: consistent random based on label hash
        else {
            java.util.Random rand = new java.util.Random(label.hashCode());
            for (int i = 0; i < 4; i++) {
                features[i] = 0.3 + rand.nextDouble() * 0.4;
            }
        }

        return features;
    }

    // =========================================================================
    // Step 3 - Contrastive Loss  (Assignment Formula)
    // =========================================================================

    /**
     * L = max(0, C - ||D(x1) - D(x2)||_2)
     *
     * L = 0    → pair already far apart, no penalty
     * L > 0    → pair too close, model is penalised
     */
    static double contrastiveLoss(double[] Dx1, double[] Dx2, double C) {
        return Math.max(0.0, C - l2Distance(Dx1, Dx2));
    }

    /** Similarity percentage: 100% = identical, 0% = maximally different */
    static double similarityPct(double[] x1, double[] x2) {
        double maxDist = Math.sqrt(x1.length);
        return (1.0 - l2Distance(x1, x2) / maxDist) * 100.0;
    }

    // =========================================================================
    // Step 4 - Main
    // =========================================================================
    public static void main(String[] args) {

        line('=', 65);
        System.out.println("  Assignment 1 -- Contrastive Learning");
        System.out.println("  Formula : L = max(0, C - ||D(x1) - D(x2)||_2)   C = " + MARGIN_C);
        System.out.println("  Course  : Algorithms");
        System.out.println("  Lecturer: Xiang-Rui Huang, NPUST / NYCU");
        line('=', 65);

        // Load all subjects from images/
        List<Subject> subjects = loadSubjects();
        if (subjects.size() < 2) {
            System.out.println("  Need at least 2 images to compare. Exiting.");
            return;
        }

        // Print embeddings table
        System.out.println("[Embeddings] D(x) for each subject\n");
        System.out.printf("  %-20s %-12s %-22s %5s %5s %5s %5s%n",
                "Name", "Label", "File", "fur", "ear", "body", "snout");
        line();
        for (Subject s : subjects) {
            System.out.printf("  %-20s %-12s %-22s %5.1f %5.1f %5.1f %5.1f%n",
                    s.name, "[" + s.label + "]", s.filename,
                    s.embedding[0], s.embedding[1],
                    s.embedding[2], s.embedding[3]);
        }

        // Pairwise comparison — ALL pairs
        int n     = subjects.size();
        int total = n * (n - 1) / 2;
        System.out.println("\n[Comparison] " + total + " pair(s) from " + n + " subjects\n");
        line();

        List<double[]> allLosses = new ArrayList<>();
        String bestPair = "", worstPair = "";
        double bestSim  = -1, worstSim = 200;

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                Subject s1 = subjects.get(i);
                Subject s2 = subjects.get(j);

                double l1   = l1Distance(s1.embedding, s2.embedding);
                double l2   = l2Distance(s1.embedding, s2.embedding);
                double sim  = similarityPct(s1.embedding, s2.embedding);
                double loss = contrastiveLoss(s1.embedding, s2.embedding, MARGIN_C);
                boolean same = s1.label.equals(s2.label);
                String ptype = same ? "Positive" : "Negative";
                String mark  = same ? "[SAME CLASS]" : "[DIFF CLASS]";

                System.out.printf("%n  %s  <->  %s%n", s1.name, s2.name);
                System.out.printf("  Labels      : %s  |  %s     Pair: %s %s%n",
                        s1.label, s2.label, ptype, mark);
                System.out.printf("  L1 distance : %.4f  [%s]%n", l1,   bar(l1 / 2.0, 28));
                System.out.printf("  L2 distance : %.4f  [%s]%n", l2,   bar(l2,       28));
                System.out.printf("  Similarity  : %5.1f%%  [%s]%n", sim, bar(sim / 100.0, 28));
                System.out.printf("  Loss  L     : %.4f  [%s]%n", loss, bar(loss,      28));

                if (!same && loss == 0.0)
                    System.out.println("  -> No penalty: already separated by >= C  (OK)");
                else if (!same)
                    System.out.printf("  -> Penalty %.4f: push these apart (different class)%n", loss);
                else if (loss == 0.0)
                    System.out.println("  -> Same class, far apart: consider pulling closer");
                else
                    System.out.println("  -> Same class, close together: correct behaviour  (OK)");

                line();
                allLosses.add(new double[]{sim, loss});

                // track best/worst similarity
                String pairLabel = s1.name + " / " + s2.name;
                if (sim > bestSim)  { bestSim  = sim;  bestPair  = pairLabel; }
                if (sim < worstSim) { worstSim = sim;  worstPair = pairLabel; }
            }
        }

        // Summary table
        System.out.println("\n[Summary] All Pairs\n");
        System.out.printf("  %-35s %5s %5s %6s %6s  %-10s%n",
                "Pair", "L1", "L2", "Sim%", "Loss", "Type");
        line();
        int idx = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                Subject s1   = subjects.get(i);
                Subject s2   = subjects.get(j);
                double  l1   = l1Distance(s1.embedding, s2.embedding);
                double  l2   = l2Distance(s1.embedding, s2.embedding);
                double  sim  = similarityPct(s1.embedding, s2.embedding);
                double  loss = contrastiveLoss(s1.embedding, s2.embedding, MARGIN_C);
                String  type = s1.label.equals(s2.label) ? "Positive" : "Negative";
                System.out.printf("  %-35s %5.2f %5.2f %5.1f%% %6.4f  %s%n",
                        s1.name + " / " + s2.name, l1, l2, sim, loss, type);
            }
        }

        // Stats
        System.out.println("\n[Stats]\n");
        line();
        System.out.printf("  Most similar pair  : %s  (%.1f%%)%n", bestPair,  bestSim);
        System.out.printf("  Least similar pair : %s  (%.1f%%)%n", worstPair, worstSim);
        System.out.println("\n  Core idea: PULL same-class pairs together,");
        System.out.println("             PUSH different-class pairs apart.");
        line('=', 65);
    }

    // =========================================================================
    // Utilities
    // =========================================================================

    static String bar(double value, int width) {
        int f = (int) Math.round(Math.min(value, 1.0) * width);
        return "#".repeat(f) + ".".repeat(width - f);
    }

    static void line(char c, int n) {
        System.out.println(String.valueOf(c).repeat(n));
    }
    static void line() { line('-', 65); }

    static String extension(String filename) {
        int dot = filename.lastIndexOf('.');
        return (dot >= 0) ? filename.substring(dot + 1) : "";
    }

    static Map<String, String> parseProperties(File file) throws IOException {
        Map<String, String> map = new LinkedHashMap<>();
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) continue;
                int eq = line.indexOf('=');
                if (eq < 0) continue;
                map.put(line.substring(0, eq).trim(), line.substring(eq + 1).trim());
            }
        }
        return map;
    }
}