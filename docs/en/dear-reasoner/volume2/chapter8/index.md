# Chapter 8: The Twilight of Rules

:::info
**Mr. Pallas's Cat's Warm Welcome**  
We spent seven chapters wandering through a deterministic universe. From the dot-dash of the telegraph to the memory palaces of dynamic programming, every step of reasoning was as precise as clockwork. But today, we face a different world — a world that is fuzzy, continuous, and full of uncertainty. Here, if-else begins to crumble, precise rules meet their challenge, and a new mode of thinking is being born at dawn.
:::

---

## Core Question: When Logic Meets Randomness, How Do Rules Evolve?

Piglet stared at the code on his computer screen, brow furrowed. "I've been trying to write a program to recognize cats in pictures, but after three days, my if-else statements have piled up like a small mountain."

It was an early summer evening in Kangle Garden at Sun Yat-sen University. A gentle breeze drifted through the arched windows of the Black Stone House, carrying the faint scent of bauhinia blossoms. The setting sun dyed the red-brick walls a warm orange, and the light in the study was soft and tranquil.

By the window, Little Seal looked up from *A Brief History of Complexity Science*. "That's a classic problem. Historically, people believed that with enough rules, you could describe anything. But in the mid-20th century, cybernetics and systems theory began to challenge that idea."

Mr. Pallas's Cat set down his gongfu tea set and smiled. "You've encountered a fundamental shift. From **deterministic rules** to **probabilistic thinking**, from **discrete logic** to **continuous representation**. Today, we explore the starting point of this transformation together."

## The Cat Dilemma: The Collapse of if-else

Piglet pointed at the dense code on the screen:

```python
if has_four_legs:
    if has_fur:
        if has_pointed_ears:
            if has_whiskers:
                if tail_is_long_and_thin:
                    return "might be a cat"
                else:
                    # but some cats have short tails
            else:
                # some cats don't have whiskers?
        else:
            # Scottish Folds don't have pointed ears
    else:
        # what about hairless cats?
else:
    # but a sitting cat only shows two legs
```

"Professor, I've written 287 condition checks," Piglet sighed. "But every time I find a new picture, I discover a new exception. This cat is in shadow, that cat only shows half its face, and there's one in a pose I've never seen before..."

Little Seal walked to the computer and studied the nested conditionals carefully. "This reminds me of the 18th-century 'mechanical philosophy' trend. People back then believed that if you found enough laws of mechanics, you could fully understand the universe. But they soon encountered chaotic systems and complex phenomena."

Mr. Pallas's Cat nodded. "Yes, this is the core challenge we face today: **why do simple rules fail when dealing with a complex world?**"

He walked to the whiteboard and drew two circles.

"The left circle is the **world describable by rules**," Mr. Pallas's Cat said. "For example: 'if the input number is greater than 10, output Yes.' The right circle is the **world hard to describe with rules** — for example, 'recognize whether a picture contains a cat.'"

Piglet studied the circles carefully. "Isn't the world describable by rules just like all those algorithms we learned in the previous seven chapters?"

"Exactly," Mr. Pallas's Cat smiled. "Linear search, binary search, greedy algorithms, dynamic programming — these all excel on problems with clear rules and sharp boundaries. But the world on the right..."

He drew many small dots inside the right circle.

"...is **unstructured**, **high-dimensional**, **full of noise and uncertainty**. In this world, rules begin to fail."

---

## The Limits of Rules: From Three Cats to Thirty Thousand Cats

Piglet thought for a moment. "So the problem is that there are too many exceptions?"

"It's not just exceptions," Little Seal picked up. "It's that **the boundaries of concepts themselves** are not clear. What is a 'cat'? Biologists have a genetic definition, zoologists have a morphological definition, and ordinary people have a visual impression. These definitions overlap, but don't fully coincide."

Mr. Pallas's Cat took a photo album from the bookshelf and turned to a page.

"Look at these three photos," he pointed. "The first is a standard orange tabby. The second is 'a cat in a cardboard box with only its tail visible.' The third is 'a cat-shaped silhouette in an abstract painting.' All three are 'cats,' but their visual features are radically different."

Piglet leaned in. "They really are! The first matches every rule I wrote. The second only matches 'has a tail.' The third... matches almost none of my rules."

"This is the **fundamental limitation of the rule-based approach**," Mr. Pallas's Cat said slowly. "Rules attempt to describe infinite variation with finite conditions. It's like trying to describe every dream with a finite vocabulary — some details will always be lost."

Little Seal pondered for a moment. "Historically, when did people first become aware of this problem?"

"The mid-20th century," Mr. Pallas's Cat said. "When early AI researchers tried to use symbolic logic to solve real-world problems, they encountered the same dilemma. This was one of the root causes of the so-called 'AI winter' — excessive optimism about rule-based methods, followed by the harsh challenge of reality."

---

## A Turning Point in Thought: From Rules to Patterns

Outside the window, the sky was growing dark, and warm lamplight filled the Black Stone House.

"Professor," Piglet asked, "if rules aren't enough, what should we do?"

Mr. Pallas's Cat picked up the gongfu tea set again and began to brew tea. Steam rose slowly in the lamplight.

"We need a **turning point in thought**," he said. "From 'seeking rules' to 'recognizing patterns.'"

He wrote two words on the whiteboard:

**Rules** → **Patterns**

"A rule is: 'if it has whiskers, it's a cat.' A pattern is: 'observe tens of thousands of cat pictures and discover their common features.'"

Piglet's eyes lit up. "That sounds like... learning?"

"Exactly," Mr. Pallas's Cat smiled. "From **rule-based systems** to **learning-based methods**. Instead of us telling the computer what a cat is, we let the computer 'learn' the features of cats from data on its own."

Little Seal mused: "This is like the evolution of scientific method — from deductive reasoning to inductive reasoning?"

"A fine analogy," Mr. Pallas's Cat nodded. "Deduction derives specific cases from general rules (all cats have whiskers → this animal has whiskers → it is a cat). Induction summarizes general patterns from specific cases (seeing many cats with whiskers → 'whiskers' may be associated with 'cat')."

---

## Orthogonal Computation Graphs: The Shift from Deterministic to Probabilistic Thinking

Mr. Pallas's Cat walked to the blackboard and drew a more detailed orthogonal computation graph.

![From Rules to Patterns Orthogonal Computation Graph](/figures/rule_to_pattern_ortho.svg)

"This graph shows the contrast between two systems," he pointed at the left dashed box. "The **deterministic rule system** — processes input through layered rules, ultimately producing a binary judgment (0 or 1)."

"And the right dashed box," he pointed to the right, "is the **probabilistic pattern system**. It first extracts features, then combines these features, and finally computes a probability output (a value between 0.0 and 1.0)."

Piglet studied the arrows in the diagram carefully. "What's this 'Transition' diamond in the middle?"

"That is the pivot point of the mental model," Mr. Pallas's Cat explained. "From **rule thinking** to **pattern thinking**. Not a simple replacement, but an expansion of our thinking tools."

Little Seal mused: "So this graph not only shows two systems, but also the transitional relationship between them?"

"Exactly," Mr. Pallas's Cat nodded. "Orthogonal computation graphs let us 'see' the transformation of mental models. Right-angle lines emphasize structural regularity; the left-to-right layout follows the evolutionary direction of thought."

"In the world of deterministic rules," he pointed at the left system, "we deal with clear, discrete judgments. If the input satisfies conditions A, B, C, the output is a definite 'yes' or 'no.'"

"But in the world of probabilistic thinking," he pointed to the right system, "we deal with **possibility**. The input goes through feature extraction, yielding a set of feature values, and then the model gives a probability — 'this picture has an 87% chance of being a cat.'"

Piglet looked at the computation graph seriously. "So we no longer pursue 100% correctness, but accept a degree of uncertainty?"

"Yes," said Mr. Pallas's Cat. "This is a compromise we must make when dealing with a complex world. We give up absolute certainty in exchange for the ability to handle fuzzy boundaries."

---

## Mental Model: Fuzzy Logic and Probabilistic Thinking

Little Seal took a book from the shelf. "Professor, this reminds me of the history of fuzzy logic and probability theory."

"Good connection," said Mr. Pallas's Cat. "In the 1960s, Lotfi Zadeh proposed fuzzy logic precisely to handle the concept of 'partial truth' — fuzzy judgments like 'somewhat hot' or 'somewhat like a cat.'"

He wrote the mental models on the whiteboard:

### Mental Model: From Binary to Continuous
- **Binary thinking**: clear distinctions, black and white opposition (traditional logic)
- **Fuzzy thinking**: matters of degree, grayscale gradation (fuzzy logic)
- **Probabilistic thinking**: distributions of possibility, quantified uncertainty (probability theory)

"These three modes of thinking," Mr. Pallas's Cat explained, "correspond to different ways of understanding the world."

Piglet thought: "So the cat-recognition problem requires fuzzy thinking or probabilistic thinking?"

"More precisely," Mr. Pallas's Cat answered, "we need **probabilistic fuzzy thinking**. We acknowledge the fuzziness of boundaries while using probability to quantify that fuzziness."

---

## Three Insights: After Twilight Comes Dawn

It was completely dark outside; the nightscape of the Pearl River glittered in the distance.

"Professor," Little Seal asked, "does the twilight of rules mean that rules have completely failed?"

Mr. Pallas's Cat shook his head. "No. After twilight comes dawn. The twilight of rules is not the death of rules, but the **evolution** of rules."

He summarized three key insights:

### First Insight: The Effective Domain of Rules
"Every rule has its **effective domain** — within this domain, it works well; beyond it, it begins to fail. The cat-recognition problem lies beyond the effective domain of simple rules."

### Second Insight: The Layering of Complexity
"The world is layered. At the bottom layer, physical laws are deterministic and rule-based. But at higher layers, phenomena such as life, consciousness, and society exhibit **emergent complexity** that is hard to describe with simple rules."

### Third Insight: The Expansion of Thinking Tools
"When old tools are insufficient, we need new ones. From rules to patterns, from logic to probability, from determinism to fuzziness — this is an expansion of thinking tools, not a replacement."

---

## Key Takeaways

:::info
**Mr. Pallas's Cat's Summary**  
1. **The boundaries of rules**: deterministic rules encounter fundamental limitations when dealing with an unstructured, high-dimensional world  
2. **The rise of patterns**: the shift from "seeking rules" to "recognizing patterns" is the crucial pivot from rule-based thinking to learning-based thinking  
3. **Probabilistic thinking**: accepting uncertainty, replacing binary judgments with quantified probability, is a vital tool for handling fuzzy boundaries  
4. **The evolution of mental models**: from binary thinking to fuzzy thinking to probabilistic thinking — reflecting different levels of understanding a complex world  
5. **Tools, not replacements**: new modes of thinking expand rather than replace old ones; each tool has its suitable domain
:::

---

## Code Practice: From Rules to Patterns in Python

"Let's use Python code to practice the shift from rule-based thinking to pattern-based thinking," said Mr. Pallas's Cat. "Code not only helps us understand the differences between the two approaches, but also lets us 'run' the transition from deterministic to probabilistic."

### Rule-Based Cat Detection: The if-else Dilemma

```python
# Rule-based cat detection system — demonstrating the limits of rules
class RuleBasedCatDetector:
    """Rule-based cat detection system"""
    
    def __init__(self):
        self.rules = [
            ("has_four_legs", 0.2),
            ("has_fur", 0.2),
            ("has_pointed_ears", 0.15),
            ("has_whiskers", 0.15),
            ("has_long_thin_tail", 0.1),
            ("makes_meow_sound", 0.1),
            ("likes_to_chase_mice", 0.05),
            ("sleeps_more_than_12_hours", 0.05)
        ]
    
    def check_rule(self, rule_name, animal_features):
        """Check a single rule"""
        return animal_features.get(rule_name, False)
    
    def detect_cat(self, animal_features):
        """Rule-based judgment of whether it's a cat"""
        # Calculate total weight of satisfied rules
        total_score = 0
        satisfied_rules = []
        
        for rule_name, weight in self.rules:
            if self.check_rule(rule_name, animal_features):
                total_score += weight
                satisfied_rules.append(rule_name)
        
        # If total score exceeds threshold, classify as cat
        threshold = 0.7  # 70% of rules satisfied
        is_cat = total_score >= threshold
        
        return is_cat, total_score, satisfied_rules

# Test the rule system
print("Rule-Based Cat Detection System Test:")
detector = RuleBasedCatDetector()

# Test case 1: typical cat
typical_cat = {
    "has_four_legs": True,
    "has_fur": True,
    "has_pointed_ears": True,
    "has_whiskers": True,
    "has_long_thin_tail": True,
    "makes_meow_sound": True,
    "likes_to_chase_mice": True,
    "sleeps_more_than_12_hours": True
}

# Test case 2: partially visible cat (e.g., cat in a box)
partial_cat = {
    "has_four_legs": False,  # only two legs visible
    "has_fur": True,
    "has_pointed_ears": False,  # ears obscured
    "has_whiskers": True,
    "has_long_thin_tail": True,
    "makes_meow_sound": False,  # quiet state
    "likes_to_chase_mice": False,
    "sleeps_more_than_12_hours": True
}

# Test case 3: dog (similar features but not a cat)
dog = {
    "has_four_legs": True,
    "has_fur": True,
    "has_pointed_ears": True,
    "has_whiskers": False,
    "has_long_thin_tail": False,  # dog tails are usually different
    "makes_meow_sound": False,
    "likes_to_chase_mice": False,
    "sleeps_more_than_12_hours": False
}

# Run tests
test_cases = [("Typical Cat", typical_cat), ("Partial Cat", partial_cat), ("Dog", dog)]

for name, features in test_cases:
    is_cat, score, rules = detector.detect_cat(features)
    print(f"\n{name}:")
    print(f"  Total score: {score:.2f}")
    print(f"  Classified as cat: {is_cat}")
    print(f"  Satisfied rules: {rules[:3]}...")  # show only first 3 rules
    
print("\nLimitations of rule-based systems:")
print("- Partial cats (in a box) may be misclassified")
print("- All rules and weights must be manually defined")
print("- Fuzzy boundary cases are hard to handle")
```

### Simple Probabilistic Classifier: From Rules to Patterns

```python
import random
from collections import defaultdict

# Simple probabilistic classifier (Naive Bayes style)
class ProbabilisticCatClassifier:
    """Probabilistic cat classifier — based on pattern recognition"""
    
    def __init__(self):
        self.cat_features = defaultdict(int)  # feature frequency for cats
        self.non_cat_features = defaultdict(int)  # feature frequency for non-cats
        self.cat_count = 0
        self.non_cat_count = 0
    
    def train(self, training_data):
        """Learn feature patterns from training data"""
        for features, is_cat in training_data:
            if is_cat:
                self.cat_count += 1
                for feature, value in features.items():
                    if value:  # only record True features
                        self.cat_features[feature] += 1
            else:
                self.non_cat_count += 1
                for feature, value in features.items():
                    if value:
                        self.non_cat_features[feature] += 1
    
    def predict_probability(self, features):
        """Predict the probability that the given features belong to a cat"""
        if self.cat_count == 0 or self.non_cat_count == 0:
            return 0.5  # neutral probability when no training data
        
        # Calculate prior probability
        p_cat = self.cat_count / (self.cat_count + self.non_cat_count)
        p_not_cat = self.non_cat_count / (self.cat_count + self.non_cat_count)
        
        # Calculate likelihood (simplified, assumes feature independence)
        likelihood_cat = 1.0
        likelihood_not_cat = 1.0
        
        for feature, value in features.items():
            if value:  # feature is True
                # Probability of feature given cat (add-1 smoothing)
                p_feature_given_cat = (self.cat_features.get(feature, 0) + 1) / (self.cat_count + 2)
                # Probability of feature given not cat
                p_feature_given_not_cat = (self.non_cat_features.get(feature, 0) + 1) / (self.non_cat_count + 2)
                
                likelihood_cat *= p_feature_given_cat
                likelihood_not_cat *= p_feature_given_not_cat
        
        # Apply Bayes' theorem
        evidence = p_cat * likelihood_cat + p_not_cat * likelihood_not_cat
        if evidence == 0:
            return 0.5
        
        probability_cat = (p_cat * likelihood_cat) / evidence
        return probability_cat

# Generate simulated training data
def generate_training_data(num_samples=1000):
    """Generate simulated training data for cats and non-cats"""
    data = []
    
    for _ in range(num_samples):
        # Randomly decide whether it's a cat
        is_cat = random.random() > 0.5
        
        features = {}
        if is_cat:
            # Cat features have higher probability
            features["has_four_legs"] = random.random() > 0.1  # 90% probability
            features["has_fur"] = random.random() > 0.05  # 95% probability
            features["has_pointed_ears"] = random.random() > 0.2  # 80% probability
            features["has_whiskers"] = random.random() > 0.1  # 90% probability
            features["makes_meow_sound"] = random.random() > 0.15  # 85% probability
        else:
            # Non-cat features (could be dogs, rabbits, etc.)
            features["has_four_legs"] = random.random() > 0.2  # 80% probability
            features["has_fur"] = random.random() > 0.3  # 70% probability
            features["has_pointed_ears"] = random.random() > 0.5  # 50% probability
            features["has_whiskers"] = random.random() > 0.7  # 30% probability
            features["makes_meow_sound"] = random.random() > 0.95  # 5% probability
        
        data.append((features, is_cat))
    
    return data

# Train and test probabilistic classifier
print("\nProbabilistic Classifier Training & Test:")
print("=" * 50)

# Generate training data
training_data = generate_training_data(500)
print(f"Generated {len(training_data)} training samples")

# Create and train classifier
classifier = ProbabilisticCatClassifier()
classifier.train(training_data)

# Test case
test_features = {
    "has_four_legs": True,
    "has_fur": True,
    "has_pointed_ears": True,
    "has_whiskers": True,
    "makes_meow_sound": True
}

probability = classifier.predict_probability(test_features)
print(f"\nTest features: {list(test_features.keys())}")
print(f"Probability of being a cat: {probability:.2%}")

# Compare different feature combinations
print("\nProbability comparison for different feature combinations:")
test_cases = [
    {"name": "Typical cat features", "features": {"has_four_legs": True, "has_fur": True, "has_pointed_ears": True, "has_whiskers": True, "makes_meow_sound": True}},
    {"name": "Partial cat features", "features": {"has_four_legs": True, "has_fur": True, "has_pointed_ears": False, "has_whiskers": True, "makes_meow_sound": False}},
    {"name": "Non-cat features", "features": {"has_four_legs": True, "has_fur": True, "has_pointed_ears": True, "has_whiskers": False, "makes_meow_sound": False}},
    {"name": "Ambiguous case", "features": {"has_four_legs": False, "has_fur": True, "has_pointed_ears": True, "has_whiskers": True, "makes_meow_sound": True}}
]

for case in test_cases:
    prob = classifier.predict_probability(case["features"])
    print(f"  {case['name']}: {prob:.2%}")
```

### Fuzzy Logic Implementation: From Binary to Continuous

```python
# Fuzzy logic implementation — handling matters of degree
class FuzzyLogicSystem:
    """Fuzzy logic system for handling matters of degree"""
    
    def __init__(self):
        # Define membership functions for fuzzy sets
        self.membership_functions = {
            "temperature": {
                "cold": lambda x: max(0, min(1, (20 - x) / 10)) if x <= 20 else 0,
                "warm": lambda x: max(0, min(1, (x - 15) / 10)) if x <= 25 else max(0, min(1, (35 - x) / 10)),
                "hot": lambda x: max(0, min(1, (x - 25) / 10)) if x >= 25 else 0
            },
            "brightness": {
                "dark": lambda x: max(0, min(1, (50 - x) / 50)),
                "medium": lambda x: max(0, min(1, abs(x - 125) / 75)),
                "bright": lambda x: max(0, min(1, (x - 100) / 100))
            }
        }
        
        # Fuzzy rule base
        self.rules = [
            # Format: (premise, conclusion, strength)
            (("temperature", "cold"), "heater_on", 0.8),
            (("temperature", "warm"), "comfortable", 0.9),
            (("temperature", "hot"), "ac_on", 0.7),
            (("brightness", "dark"), "lights_on", 0.9),
            (("brightness", "medium"), "lights_adjust", 0.6),
            (("brightness", "bright"), "lights_off", 0.8)
        ]
    
    def fuzzify(self, variable_name, value):
        """Fuzzify a precise value"""
        result = {}
        if variable_name in self.membership_functions:
            for category, func in self.membership_functions[variable_name].items():
                membership = func(value)
                if membership > 0:
                    result[category] = membership
        return result
    
    def apply_rules(self, inputs):
        """Apply fuzzy rules"""
        # Fuzzify all inputs
        fuzzy_sets = {}
        for var_name, value in inputs.items():
            fuzzy_sets[var_name] = self.fuzzify(var_name, value)
        
        # Apply rules
        rule_outputs = {}
        for (var_name, category), conclusion, strength in self.rules:
            if var_name in fuzzy_sets and category in fuzzy_sets[var_name]:
                membership = fuzzy_sets[var_name][category]
                activation = min(membership, strength)
                
                if conclusion not in rule_outputs:
                    rule_outputs[conclusion] = activation
                else:
                    rule_outputs[conclusion] = max(rule_outputs[conclusion], activation)
        
        return rule_outputs
    
    def defuzzify(self, rule_outputs, method="centroid"):
        """Defuzzify (return the most likely action)"""
        if not rule_outputs:
            return "no_action"
        
        # Simple method: choose the conclusion with highest activation
        if method == "max":
            return max(rule_outputs.items(), key=lambda x: x[1])[0]
        
        # Weighted average
        elif method == "weighted_average":
            total_weight = sum(rule_outputs.values())
            if total_weight == 0:
                return "no_action"
            
            # Simple mapping to numerical values
            action_values = {
                "heater_on": 1,
                "comfortable": 2,
                "ac_on": 3,
                "lights_on": 4,
                "lights_adjust": 5,
                "lights_off": 6,
                "no_action": 0
            }
            
            weighted_sum = sum(action_values.get(action, 0) * weight 
                              for action, weight in rule_outputs.items())
            
            avg_value = weighted_sum / total_weight
            
            # Find the closest action
            closest_action = min(action_values.items(), 
                                key=lambda x: abs(x[1] - avg_value))[0]
            return closest_action

# Fuzzy logic system demonstration
print("\nFuzzy Logic System Demonstration:")
print("=" * 50)

fuzzy_system = FuzzyLogicSystem()

# Test different temperature and brightness combinations
test_scenarios = [
    {"temperature": 10, "brightness": 30},   # cold and dark
    {"temperature": 22, "brightness": 150},  # warm and bright
    {"temperature": 30, "brightness": 80},   # hot and medium brightness
    {"temperature": 18, "brightness": 200}   # cool and very bright
]

for scenario in test_scenarios:
    print(f"\nScenario: Temp={scenario['temperature']}°C, Brightness={scenario['brightness']}lux")
    
    # Fuzzify
    temp_fuzzy = fuzzy_system.fuzzify("temperature", scenario["temperature"])
    bright_fuzzy = fuzzy_system.fuzzify("brightness", scenario["brightness"])
    
    print(f"  Temperature fuzzy sets: {temp_fuzzy}")
    print(f"  Brightness fuzzy sets: {bright_fuzzy}")
    
    # Apply rules
    outputs = fuzzy_system.apply_rules(scenario)
    print(f"  Rule outputs: {outputs}")
    
    # Defuzzify
    action = fuzzy_system.defuzzify(outputs, method="weighted_average")
    print(f"  Final action: {action}")
```

---

## Mr. Pallas's Cat's Reflection Questions

### Hands-On Exploration (for Piglet)
1. **Rule experiment**: try to describe "a smile" using if-else rules. How many rules do you need? What difficulties do you encounter?
2. **Probability exercise**: design a simple probabilistic classifier. Input: brightness value of an image. Output: probability of "daytime" vs. "nighttime."
3. **Boundary exploration**: find a problem you think can be perfectly described by rules, and another that absolutely cannot. Analyze the essential difference between them.

### Historical Investigation (for Little Seal)
1. **Tracing intellectual history**: track the development from Aristotelian logic to fuzzy logic. Which thinkers played key roles?
2. **AI winter research**: study the "AI winter" of the 1970s-80s. What role did the limitations of rules play in it?
3. **Cross-disciplinary connections**: how are fuzzy logic and probability theory applied in engineering, medicine, and economics?

### Integrated Reflection
1. **Philosophical reflection**: if the world is fundamentally probabilistic, how must the concept of "truth" be redefined?
2. **Ethical challenge**: in medical diagnosis using probabilistic AI systems, how should doctors and patients understand "87% probability of malignancy"?
3. **Creative exercise**: design a "fuzzy rule" system to describe "a beautiful sunset." How do you quantify the degree of "beauty"?

---

## Coming Up Next

The fragrance of tea filled the Black Stone House; night had fallen deep.

"Today we witnessed the twilight of rules," said Mr. Pallas's Cat. "In the next chapter, we welcome a new dawn: **from discrete to continuous**."

Piglet asked curiously: "Continuous? Like how real numbers can be infinitely subdivided?"

"Not just mathematical continuity," Mr. Pallas's Cat explained. "It's a continuity of thought. What if logic were no longer 0 and 1, but an infinite spectrum of possibilities between 0 and 1?"

Little Seal flipped through his notebook. "This introduces the concepts of vector spaces and continuous representation. Historically, how did people arrive at this direction?"

Mr. Pallas's Cat smiled. "We'll take it slow. See you in the next chapter."

---

> **Piglet's note**: I tried to describe "a smile" with rules. After writing 50 if-else statements, I gave up. But using a probabilistic approach — analyzing the features of 1,000 smiling-face images — actually yielded decent results. Sometimes, giving up control is the path to understanding.
>
> **Little Seal's note**: I researched the history of fuzzy logic and was surprised to find its earliest applications in control engineering (e.g., subway braking systems). The most abstract mathematical ideas often blossom in the most practical places.
>
> **Mr. Pallas's Cat's closing words**: The twilight of rules is not an end, but the beginning of a new mode of thinking. On this fuzzy boundary, we learn humility — acknowledging the limits of cognition, and then seeking ways to transcend those limits. This is the true spirit of science.
