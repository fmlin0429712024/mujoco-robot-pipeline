# Robot Learning Methods Overview

> **Learn the landscape:** Understand when robots need learning vs when they don't

**Reading time:** 5 minutes

---

## The Core Question

**When building a robot, should it learn or just follow programmed rules?**

This is the first decision you'll make, and it determines your entire approach.

---

## Three Approaches to Robot Intelligence

```
Classical Methods          Learning Methods           Hybrid Approach
(No Training)              (Needs Training)          (Best of Both)
     │                          │                         │
     ├─ A* pathfinding         ├─ IL: Copy expert       ├─ Classical nav
     ├─ SLAM mapping           ├─ RL: Trial & error     │  + Vision learning
     └─ PID control            └─ End-to-end DL         └─ Most real products
```

---

## When to Use Each Method

### **Classical Methods (No Learning)**

**What:** Algorithms you code once, work forever

**Examples:**
- A* for path planning  
- SLAM for mapping
- PID for motor control

**Use when:**
- ✅ Problem has known solution (navigation)
- ✅ Safety-critical (nursing homes!)
- ✅ Need immediate deployment

**Business value:** Zero training time, predictable, interpretable

---

### **Imitation Learning (IL)**

**What:** Robot learns by copying expert demonstrations (supervised learning)

**How it works:**
```
1. Expert demonstrates task (50-500 times)
2. Record states + actions
3. Train model to mimic expert
4. Deploy learned policy
```

**Use when:**
- ✅ Have expert demonstrations
- ✅ Complex manipulation tasks
- ✅ Need human-like behavior

**Example:** Pick-and-place arm (this project!)

**Business value:** Faster than RL, works for tasks too complex to hand-code

---

### **Reinforcement Learning (RL)**

**What:** Robot discovers strategy through trial and error (trial-and-error learning)

**How it works:**
```
1. Agent tries random actions
2. Gets rewards/penalties
3. Learns which actions work best
4. Eventually masters task (millions of attempts)
```

**Use when:**
- ✅ No expert available
- ✅ Need to discover optimal strategy
- ✅ Can afford long training time

**Example:** Game playing, novel locomotion

**Business value:** Can discover creative solutions, but expensive to train

---

## Decision Framework

```
Your Robot Task
      ↓
      ┌─────────────────────────────────────┐
      │ Can you solve it with an algorithm? │
      └─────────────────────────────────────┘
             ↓ Yes              ↓ No
      ┌──────────┐        ┌────────────┐
      │CLASSICAL │        │  LEARNING  │
      │(Instant!)│        │(Train first)│
      └──────────┘        └────────────┘
                                ↓
                    ┌─────────────────────┐
                    │ Have expert demos?  │
                    └─────────────────────┘
                       ↓ Yes      ↓ No
                    ┌─────┐   ┌──────┐
                    │  IL │   │  RL  │
                    │Hours│   │Weeks │
                    └─────┘   └──────┘
```

---

## Special Case: SLAM

**SLAM = Simultaneous Localization And Mapping**

**Common misconception:** SLAM is NOT a learning method!

**Reality:** SLAM is a classical algorithm that:
- Builds a map while robot explores
- Tracks robot's position on that map
- Uses geometry and probability (no neural networks!)

**Use case:** Unknown environment navigation (your AMR in nursing homes)

---

## Comparison Table

| Method | Training | Data Needed | Complexity | Safety | Use Case |
|--------|----------|-------------|-----------|--------|----------|
| **Classical** | None | Zero | Low ⭐ | ✅✅✅ | Navigation, known problems |
| **IL** | Hours-days | Medium (50-500 demos) | Medium ⭐⭐ | ✅✅ | Manipulation, human-like |
| **RL** | Days-weeks | Huge (millions of attempts) | High ⭐⭐⭐ | ⚠️ | Optimization, discovery |

---

## Key Takeaways

1. **Start simple:** Can classical methods solve it? Try them first!

2. **Learning is powerful but expensive:** Only use when necessary

3. **SLAM is classical, not learning:** Common beginner confusion

4. **IL vs RL:**
   - Have expert? → Use IL
   - No expert? → Use RL (if you can afford the training time)

5. **Real products use hybrid:**
   - Classical for navigation (reliable!)
   - Learning for perception/manipulation (flexible!)

---

## What's Next?

Now that you understand the methods, let's see how they apply to real business problems.

**→ Continue to:** [02 - Robotics Use Cases and Solutions](02_robotics_use_cases_and_solutions.md)

---

*Part 1 of 5-part learning journey*
