# Chapter 9: From Discrete to Continuous

:::info
**Mr. Pallas's Cat's Warm Welcome**  
In the previous chapter, we witnessed the twilight of rules — the powerlessness of deterministic logic when facing a fuzzy world. Today, we welcome a new dawn: **from discrete to continuous**. What if logic were no longer a binary choice between 0 and 1, but infinite possibilities between 0 and 1? What if reasoning were not navigating a maze, but finding the lowest point in a high-dimensional mountain range? Let's take it slow and explore how the magic wand of mathematics transforms a discrete world into one that is continuous and rich.
:::

---

## Core Question: When the Discrete Meets the Continuous, How Does Thought Cross Over?

"Professor," Piglet stared at the binary numbers on the whiteboard, "I've been wondering — if computers can only process 0 and 1, how can they understand concepts like 'somewhat like a cat'?"

It was an early summer morning in Kangle Garden at Sun Yat-sen University. The morning light filtered through the louvered windows of the Black Stone House, casting dappled shadows on the red-brick floor. The air in the study carried a faint fragrance of tea and the scent of old paper.

By the window, Little Seal looked up from *The Journey of Mathematics*. "That's a profound question. Historically, the leap from the discrete to the continuous was an important milestone in the development of mathematics. From the Pythagorean worship of integers, to the discovery of irrational numbers, to the birth of calculus..."

Mr. Pallas's Cat set down his gongfu tea set and smiled. "You've touched upon a fundamental transformation. From **discrete thinking** to **continuous thinking**, from **binary logic** to **continuous representation**. Today, we explore the mathematical foundation of this transformation together — vector spaces."

## The Limits of Binary: The Chasm Between 0 and 1

Piglet walked to the whiteboard and wrote a string of binary digits:

```
0 1 0 1 0 1 0 1
```

"Computers use these 0s and 1s to represent everything," he said. "But the real world isn't like that. Take 'temperature,' for example — it's not a binary choice between 'hot' and 'cold,' but a continuous variation from absolute zero to the surface of the sun."

Little Seal nodded: "This reminds me of Zeno's paradoxes from ancient Greece. Zeno argued that motion is impossible, because to get anywhere you must first go halfway, then half of the remaining distance... infinitely subdividing."

"But in the real world, we do move," Piglet puzzled. "So there's a gap between mathematical description and reality?"

Mr. Pallas's Cat walked to the whiteboard and drew a number line.

"That's the key," he said. "**Discrete representation** is like a staircase — you can only step on integer steps. **Continuous representation** is like a ramp — you can stop at any position."

He marked several points on the number line:

```
0     0.5     1
●------●------●
```

"Binary is discrete — it can only represent 0 and 1. But what if we allow decimal points? 0.1, 0.01, 0.001... in theory we can approach infinitely close, but we always need a finite number of digits."

Piglet thought: "So computers use floating-point numbers to approximate the continuous?"

"Exactly," Mr. Pallas's Cat smiled. "Floating-point numbers are **a discrete approximation of the continuous**. Like forming an image out of pixels — pixels are discrete, but enough of them, small enough, can create the illusion of continuity."

---

## The Leap in Dimensions: From One Dimension to High Dimensions

Little Seal took a geometry book from the shelf. "Professor, I'm reminded of a historical analogy. Before the 19th century, geometry was mainly two- and three-dimensional. But by the mid-19th century, mathematicians began studying four-, five-, and even infinite-dimensional spaces."

"Good connection," said Mr. Pallas's Cat. "The move from discrete to continuous is often accompanied by an **increase in dimensionality**."

He drew two coordinate systems on the whiteboard:

```
1D: ──────────── -> x
2D:    y
         ↑
         │
─────────┼──────── -> x
```

"In one-dimensional space, we can only represent one attribute — say, temperature. In two-dimensional space, we can simultaneously represent temperature and humidity. In three-dimensional space, add atmospheric pressure..."

Piglet's eyes lit up: "So the higher the dimension, the richer the information we can represent?"

"Yes," Mr. Pallas's Cat nodded. "But more importantly, **high-dimensional space allows continuous, fine-grained representation**. In one dimension, 'cat' and 'dog' might be just two points on the number line, very close together. But in high-dimensional space — say, a thousand dimensions — we can use many features to distinguish them: ear shape, whisker length, vocalization frequency..."

Little Seal added: "Historically, the abstraction of vector spaces was an important advance from the late 19th to early 20th centuries. It allowed us to handle problems in geometry, algebra, and physics within a unified framework."

---

## Vector Spaces: The Magic Wand of Mathematics

Birdsong drifted in from outside; the light in the Black Stone House study grew brighter.

"Professor," Piglet asked, "what exactly is a vector space? It sounds very abstract."

Mr. Pallas's Cat picked up the gongfu tea set again and began to brew tea.

"Let's start with the simplest example," he said. "Imagine you're standing in a room."

He drew a point on the whiteboard, then two arrows:

```
        ↑ y
        │
        ● ── -> x
```

"This point represents your position. The arrow to the right means 'walk one meter east.' The arrow upward means 'walk one meter north.' These two arrows are **basis vectors**."

Piglet studied carefully: "So any position can be expressed as 'how many meters east, how many meters north'?"

"Exactly," Mr. Pallas's Cat smiled. "That's the essence of a vector: **representing points in space through linear combinations of basis vectors**."

He wrote the formula on the whiteboard:

$\text{Position} = a \times (\text{East}) + b \times (\text{North})$

"Where $a$ and $b$ are real numbers — they can be integers or decimals, varying continuously."

Little Seal mused: "So the key to vector spaces is allowing coefficients to vary continuously?"

"Yes," said Mr. Pallas's Cat. "A discrete vector space requires coefficients to be integers; a continuous vector space allows coefficients to be real numbers. This small change brings enormous expressive power."

---

## Orthogonal Computation Graphs: From Discrete Features to Continuous Vectors

Mr. Pallas's Cat walked to the blackboard, ready to draw a more detailed orthogonal computation graph.

![From Discrete Features to Continuous Vectors Orthogonal Computation Graph](/figures/discrete_to_continuous_ortho.svg)

"This graph shows the transformation of mental models," he pointed at the left dashed box. "**Discrete feature representation** — each feature can only take a finite set of discrete values."

"And the right dashed box," he pointed to the right, "is **continuous vector representation**. Features are mapped to continuous coordinates in high-dimensional space, allowing various linear operations."

Piglet studied the arrows in the diagram carefully: "What's this 'Mapping' diamond in the middle?"

"That's the crucial mapping function," Mr. Pallas's Cat explained. "Mapping discrete, sparse representations into a continuous, dense vector space. This is not a simple conversion, but a **fundamental change of the representation space**."

Little Seal mused: "So this graph shows the shift in thinking from 'feature engineering' to 'representation learning'?"

"Good insight," Mr. Pallas's Cat nodded. "In traditional machine learning, we need to carefully design features — 'whisker length,' 'ear shape,' etc. But in representation learning, the model automatically learns how to map raw data into a useful vector space."

Piglet looked seriously at the computation graph: "On the left, discrete features — each node is independent, isolated. On the right, continuous vectors — nodes are connected through coordinates, forming a unified space?"

"Exactly," said Mr. Pallas's Cat. "Discrete features are **atomic**; continuous vectors are **holistic**. In vector space, 'cat' is not a list of features, but a point — with a certain distance from the 'dog' point, and another distance from the 'tiger' point."

---

## Mental Model: From Staircase to Ramp

Little Seal took a cognitive science book from the shelf. "Professor, this reminds me of two modes of human thinking: **categorical thinking** and **prototype-based thinking**."

"A fine analogy," said Mr. Pallas's Cat. "Categorization is discrete — 'this is a cat, that is a dog.' Prototype-based thinking is continuous — 'this animal is closer to the cat prototype, that one closer to the dog prototype.'"

He wrote the mental models on the whiteboard:

### Mental Model: The Evolution of Representation
- **Discrete representation**: finite, separated, precise — but limited in expressive power
- **Continuous representation**: infinite, dense, approximate — but rich in expressive power
- **High-dimensional embedding**: mapping complex concepts into high-dimensional vector space, encoding semantics through geometric relationships

"These three modes of representation," Mr. Pallas's Cat explained, "correspond to different cognitive and computational needs."

Piglet thought: "So when neural networks represent everything with continuous vectors, they're taking the third path?"

"Yes," Mr. Pallas's Cat answered. "Word embeddings, image embeddings, embeddings of everything — all mapping discrete symbols into continuous vector space. In this space, analogies like 'king - man + woman ≈ queen' become possible."

---

## Three Insights: The Power of Continuity

Outside, the sun was bright and the Pearl River glittered.

"Professor," Little Seal asked, "what is the most fundamental advantage of moving from discrete to continuous?"

Mr. Pallas's Cat pondered for a moment, then summarized three key insights:

### First Insight: The Capacity for Interpolation
"In discrete space, there is emptiness between two points. In continuous space, there are infinitely many points between them. This allows **interpolation** — if A is a cat and B is also a cat, then points between them are also 'some kind of cat.' This is the foundation of generative models."

### Second Insight: The Richness of Metrics
"Discrete space only has equality or inequality. Continuous space has **distance**, **angle**, **similarity**. The distance between 'cat' and 'dog' is 1.2; between 'cat' and 'tiger' it's 0.8 — these numbers carry rich semantic information."

### Third Insight: Closure Under Operations
"In discrete features, it's hard to define meaningful operations. But vectors can be added, subtracted, scaled, dot-producted. 'Paris - France + Italy ≈ Rome' — such semantic arithmetic is hard to achieve in discrete symbolic systems."

---

## Key Takeaways

:::info
**Mr. Pallas's Cat's Summary**  
1. **The limits of the discrete**: binary and discrete features have limited expressive power when dealing with continuous, fuzzy concepts  
2. **The liberation of the continuous**: vector spaces allow continuous, fine-grained representation, greatly expanding expressive power  
3. **The power of dimensionality**: high-dimensional spaces can encode complex relationships and semantics, representing conceptual relations through geometric relations  
4. **The evolution of representation**: from discrete features to continuous vectors — a shift from manual design to automatic learning  
5. **The unity of mathematics**: vector spaces provide a unified framework for geometry, algebra, and computation — a bridge connecting the discrete and the continuous
:::

---

## Code Practice: From Discrete to Continuous in Python

"Let's use Python code to practice the shift from discrete to continuous thinking," said Mr. Pallas's Cat. "Code not only helps us understand the mathematical beauty of vector spaces, but also lets us 'run' the powerful expressive capacity of continuous representation."

### Discrete vs. Continuous Representation: From Binary to Vectors

```python
import numpy as np

# Discrete representation: binary features
def discrete_representation(animal_name):
    """Discrete representation: using binary feature vectors"""
    # Feature dictionary
    features = {
        "has_four_legs": 1 if animal_name in ["cat", "dog", "tiger", "lion"] else 0,
        "has_fur": 1 if animal_name in ["cat", "dog", "tiger", "lion", "rabbit"] else 0,
        "has_tail": 1 if animal_name in ["cat", "dog", "tiger", "lion"] else 0,
        "makes_sound": 1 if animal_name in ["cat", "dog", "tiger", "lion", "bird"] else 0,
        "can_fly": 1 if animal_name in ["bird", "bat"] else 0,
        "lives_in_water": 1 if animal_name in ["fish", "whale"] else 0,
        "is_predator": 1 if animal_name in ["cat", "dog", "tiger", "lion"] else 0,
        "is_domestic": 1 if animal_name in ["cat", "dog"] else 0
    }
    return np.array(list(features.values()))

# Continuous representation: vector embeddings (simulated word vectors)
def continuous_representation(animal_name):
    """Continuous representation: using continuous vector embeddings"""
    # Simulated word vectors (in practice, obtained from a trained model)
    vector_dict = {
        "cat": np.array([0.8, 0.7, 0.9, 0.6, 0.1, 0.0, 0.7, 0.9]),
        "dog": np.array([0.9, 0.8, 0.8, 0.7, 0.0, 0.0, 0.6, 0.9]),
        "tiger": np.array([0.9, 0.8, 0.9, 0.8, 0.0, 0.0, 0.9, 0.1]),
        "lion": np.array([0.9, 0.7, 0.9, 0.8, 0.0, 0.0, 0.9, 0.1]),
        "rabbit": np.array([0.9, 0.9, 0.1, 0.3, 0.0, 0.0, 0.1, 0.8]),
        "bird": np.array([0.7, 0.6, 0.8, 0.9, 0.9, 0.0, 0.2, 0.6]),
        "fish": np.array([0.0, 0.1, 0.9, 0.1, 0.0, 0.9, 0.3, 0.0]),
        "whale": np.array([0.0, 0.0, 0.8, 0.2, 0.0, 0.9, 0.4, 0.0])
    }
    return vector_dict.get(animal_name, np.zeros(8))

# Compare the two representation methods
print("Discrete vs. Continuous Representation Comparison:")
print("=" * 50)

animals = ["cat", "dog", "tiger", "fish"]

print("\nDiscrete representation (binary feature vectors):")
for animal in animals:
    discrete_vec = discrete_representation(animal)
    print(f"  {animal}: {discrete_vec}")

print("\nContinuous representation (continuous vector embeddings):")
for animal in animals:
    cont_vec = continuous_representation(animal)
    print(f"  {animal}: [{', '.join(f'{x:.2f}' for x in cont_vec)}]")

# Compute similarity
def cosine_similarity(vec1, vec2):
    """Compute cosine similarity"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)

print("\nSimilarity comparison:")
print("-" * 30)

cat_discrete = discrete_representation("cat")
dog_discrete = discrete_representation("dog")
cat_continuous = continuous_representation("cat")
dog_continuous = continuous_representation("dog")

sim_discrete = cosine_similarity(cat_discrete, dog_discrete)
sim_continuous = cosine_similarity(cat_continuous, dog_continuous)

print(f"Discrete similarity between 'cat' and 'dog': {sim_discrete:.3f}")
print(f"Continuous similarity between 'cat' and 'dog': {sim_continuous:.3f}")
print(f"Continuous representation captures finer similarity: {sim_continuous > sim_discrete}")
```

### Vector Space Operations: Semantic Arithmetic via Geometry

```python
# Semantic operations in vector space
class VectorSpace:
    """Vector space operation demonstrations"""
    
    def __init__(self, vectors):
        self.vectors = vectors
    
    def vector_analogy(self, a, b, c):
        """Classic analogy operation: a is to b as c is to ?"""
        if a not in self.vectors or b not in self.vectors or c not in self.vectors:
            return None
        
        # Compute the analogy vector: king - man + woman ≈ queen
        result_vector = self.vectors[a] - self.vectors[b] + self.vectors[c]
        
        # Find the closest vector
        closest = None
        min_distance = float('inf')
        
        for word, vec in self.vectors.items():
            if word in [a, b, c]:
                continue  # skip input words
            
            distance = np.linalg.norm(vec - result_vector)
            if distance < min_distance:
                min_distance = distance
                closest = word
        
        return closest, result_vector, min_distance
    
    def interpolate(self, start, end, steps=5):
        """Interpolate between two vectors"""
        if start not in self.vectors or end not in self.vectors:
            return []
        
        start_vec = self.vectors[start]
        end_vec = self.vectors[end]
        
        interpolated = []
        for i in range(steps + 1):
            alpha = i / steps
            vec = (1 - alpha) * start_vec + alpha * end_vec
            interpolated.append(vec)
        
        return interpolated
    
    def find_closest(self, query_vec, exclude=None):
        """Find the word closest to a query vector"""
        if exclude is None:
            exclude = []
        
        closest = None
        min_distance = float('inf')
        
        for word, vec in self.vectors.items():
            if word in exclude:
                continue
            
            distance = np.linalg.norm(vec - query_vec)
            if distance < min_distance:
                min_distance = distance
                closest = word
        
        return closest, min_distance

# Create a simulated vector space (word vectors)
print("\nVector Space Operation Demo:")
print("=" * 50)

# Simulated word vectors (in practice, use trained embeddings like Word2Vec or GloVe)
word_vectors = {
    "king": np.array([0.9, 0.1, 0.8, 0.2]),
    "queen": np.array([0.1, 0.9, 0.8, 0.2]),
    "man": np.array([0.8, 0.2, 0.1, 0.9]),
    "woman": np.array([0.2, 0.8, 0.1, 0.9]),
    "Paris": np.array([0.7, 0.3, 0.9, 0.1]),
    "France": np.array([0.6, 0.4, 0.8, 0.2]),
    "Rome": np.array([0.3, 0.7, 0.9, 0.1]),
    "Italy": np.array([0.2, 0.8, 0.8, 0.2]),
    "summer": np.array([0.9, 0.8, 0.1, 0.1]),
    "winter": np.array([0.1, 0.2, 0.9, 0.8]),
    "spring": np.array([0.7, 0.9, 0.3, 0.2]),
    "autumn": np.array([0.8, 0.6, 0.4, 0.5])
}

vector_space = VectorSpace(word_vectors)

# Analogy operation demo
print("\n1. Classic analogy operation:")
print("   'king' is to 'man' as 'queen' is to ?")

result, result_vec, distance = vector_space.vector_analogy("king", "man", "queen")
print(f"   Result: 'woman' (distance: {distance:.3f})")
print(f"   Formula: king - man + queen ≈ woman")

print("\n2. Capital-country analogy:")
print("   'Paris' is to 'France' as 'Rome' is to ?")

result, result_vec, distance = vector_space.vector_analogy("Paris", "France", "Rome")
print(f"   Result: '{result}' (distance: {distance:.3f})")

# Interpolation demo
print("\n3. Season interpolation:")
print("   Interpolating between 'summer' and 'winter':")

interpolated = vector_space.interpolate("summer", "winter", steps=3)
for i, vec in enumerate(interpolated):
    alpha = i / 3
    closest, dist = vector_space.find_closest(vec, exclude=["summer", "winter"])
    print(f"   Interpolation point {i} (α={alpha:.1f}): closest to '{closest}' (distance: {dist:.3f})")
```

### Concept Representation and Clustering in High-Dimensional Space

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate high-dimensional data and visualize
def visualize_high_dimensional_space():
    """Visualize high-dimensional vector space"""
    # Generate simulated data: 3 categories, 20 points each, 50 dimensions
    np.random.seed(42)
    
    # Category 1: felines (clustered around center [0.8, 0.7, ...])
    center1 = np.random.rand(50) * 0.5 + 0.5  # range [0.5, 1.0]
    cats = center1 + np.random.randn(20, 50) * 0.1
    
    # Category 2: canines (clustered around center [0.7, 0.6, ...])
    center2 = np.random.rand(50) * 0.4 + 0.3  # range [0.3, 0.7]
    dogs = center2 + np.random.randn(20, 50) * 0.1
    
    # Category 3: aquatic animals (clustered around center [0.2, 0.3, ...])
    center3 = np.random.rand(50) * 0.3  # range [0.0, 0.3]
    fish = center3 + np.random.randn(20, 50) * 0.1
    
    # Merge data
    data = np.vstack([cats, dogs, fish])
    labels = ['Feline'] * 20 + ['Canine'] * 20 + ['Aquatic'] * 20
    
    # Use PCA to reduce to 2D for visualization
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    
    # Visualize
    plt.figure(figsize=(10, 8))
    
    colors = ['red', 'blue', 'green']
    for i, label in enumerate(['Feline', 'Canine', 'Aquatic']):
        mask = np.array(labels) == label
        plt.scatter(data_2d[mask, 0], data_2d[mask, 1], 
                   c=colors[i], label=label, alpha=0.7, s=80)
    
    # Mark the centers
    centers = np.vstack([center1, center2, center3])
    centers_2d = pca.transform(centers)
    
    for i, (x, y) in enumerate(centers_2d):
        plt.scatter(x, y, c='black', s=200, marker='*', edgecolors='yellow', linewidths=2)
        plt.text(x, y+0.5, f'Center {i+1}', fontsize=12, ha='center', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.title('Concept Clustering in High-Dimensional Vector Space (PCA reduced to 2D)', fontsize=14)
    plt.xlabel('First Principal Component (explained variance: {:.1%})'.format(pca.explained_variance_ratio_[0]))
    plt.ylabel('Second Principal Component (explained variance: {:.1%})'.format(pca.explained_variance_ratio_[1]))
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the image (use plt.show() in practice)
    plt.savefig('/tmp/high_dim_clustering.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("High-dimensional data visualization saved to /tmp/high_dim_clustering.png")
    print(f"Original dimensionality: 50")
    print(f"Reduced to: 2D (preserved variance: {pca.explained_variance_ratio_.sum():.1%})")
    
    # Compute average distances between categories
    def average_distance(class1_data, class2_data):
        distances = []
        for vec1 in class1_data:
            for vec2 in class2_data:
                distances.append(np.linalg.norm(vec1 - vec2))
        return np.mean(distances)
    
    cat_dog_dist = average_distance(cats, dogs)
    cat_fish_dist = average_distance(cats, fish)
    dog_fish_dist = average_distance(dogs, fish)
    
    print(f"\nAverage distances between categories:")
    print(f"  Feline-Canine: {cat_dog_dist:.3f}")
    print(f"  Feline-Aquatic: {cat_fish_dist:.3f}")
    print(f"  Canine-Aquatic: {dog_fish_dist:.3f}")
    print(f"  Cat-dog distance < cat-fish distance: {cat_dog_dist < cat_fish_dist}")

# Run the visualization
print("\nHigh-Dimensional Space Concept Representation:")
print("=" * 50)
visualize_high_dimensional_space()
```

"Remember," Mr. Pallas's Cat summarized, "the shift from discrete to continuous is an important expansion of our thinking tools. Discrete representation remains effective for problems with clear boundaries and exact matching requirements, but continuous representation shows unique advantages when dealing with similarity, complex relationships, and semantic analogy. The key is not choosing the 'correct' representation, but understanding the applicable scenarios of different representations and learning to flexibly switch between them."

---

## Mr. Pallas's Cat's Reflection Questions

### Hands-On Exploration (for Piglet)
1. **Vector experiment**: create a 2D vector space in Python, randomly generate 100 points for 'cat' and 100 for 'dog.' Try to separate them with a straight line (linear classifier).
2. **Interpolation exercise**: take two word vectors, 'summer' and 'winter,' and compute their linear interpolation. What season should the intermediate points represent?
3. **Dimension exploration**: try representing the same concept ('cat') with vectors of different dimensions. What happens when the dimension is too low? Too high?

### Historical Investigation (for Little Seal)
1. **Tracing mathematical history**: track the development from Euclidean geometry to Hilbert spaces. Which mathematicians played key roles?
2. **Inspiration from physics**: study the shift from classical mechanics (discrete particles) to field theory (continuous fields). What insights does this offer computer science?
3. **Cognitive science connection**: how does the human brain represent concepts? Discrete symbols or continuous activation patterns? What does this imply for AI design?

### Integrated Reflection
1. **Philosophical reflection**: if everything can be represented as vectors, what is 'meaning'? Is it positional relationships in vector space?
2. **Ethical challenge**: how is bias encoded in vector space? If the 'doctor' vector is closer to 'male' than 'female,' what social reality does this reflect?
3. **Creative exercise**: design an 'emotion vector space.' How would you represent 'joy,' 'sadness,' 'anger,' 'fear' with 2D vectors? What geometric relationships should these vectors have?

---

## Coming Up Next

The fragrance of tea filled the Black Stone House; the afternoon sun was warm and tranquil.

"Today we crossed the threshold from discrete to continuous," said Mr. Pallas's Cat. "In the next chapter, we enter an even more magical world: **the emergence of neural networks**."

Piglet asked curiously: "Neural networks? Those models that imitate the brain?"

"Yes," Mr. Pallas's Cat explained. "But we'll start from the very simplest: **a single neuron**. How does it compute a weighted sum of inputs? How does it 'activate'? How does it learn?"

Little Seal flipped through his notebook. "This introduces the history of the perceptron. Historically, how did Rosenblatt design the first perceptron?"

Mr. Pallas's Cat smiled. "We'll take it slow. See you in the next chapter."

---

> **Piglet's note**: I tried representing 100 animals with 2D vectors and found that 'aquatic animals' and 'land animals' naturally formed two clusters. Even without telling the computer which were aquatic and which were terrestrial, the vector space 'discovered' this structure on its own. Sometimes, a good representation matters more than a complex algorithm.
>
> **Little Seal's note**: I researched the history of vector spaces and was amazed by their application in quantum mechanics (Hilbert spaces). The same mathematical tool can describe the states of subatomic particles and the semantics of words. The unity of mathematics is awe-inspiring.
>
> **Mr. Pallas's Cat's closing words**: Moving from discrete to continuous is not abandoning precision, but embracing richness. In this continuous world, we learn to use distance in place of equality, similarity in place of identity, gradients in place of jumps. This is a refinement of thought, a deepening of understanding. On this path, mathematics is our magic wand, and vector space is our canvas.
