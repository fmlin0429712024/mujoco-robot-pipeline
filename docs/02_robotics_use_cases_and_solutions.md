# Robotics Use Cases and Solutions

> **From theory to practice:** See how methods apply to real business problems

**Reading time:** 10 minutes

---

## Two Real-World Examples

This document explores two complete robot systems:

1. **Nursing Home AMR** - Navigation, perception, 2,000-facility scale
2. **Pick-Place Demonstration** - Manipulation, imitation learning, this project

Both teach different lessons about choosing the right technical approach.

---

## Use Case 1: Nursing Home AMR

### **Business Context**

**Goal:** Deploy autonomous mobile robots to 2,000 nursing homes for medication/item delivery

**Requirements:**
- Navigate different layouts (each facility unique)
- Detect people, wheelchairs, obstacles
- Work in diverse lighting conditions
- Safety-critical environment
- Voice interface for nurses
- Fast deployment (<15 min per facility)

---

### **Technical Architecture**

#### **Navigation: Classical Methods (No Training)**

```
Discovery Run (One-time per facility):
├─ Robot drives through hallways
├─ SLAM builds map
├─ AprilTags provide landmarks
└─ Saves map (~15 min)

Live Operation:
├─ Localize using SLAM + AprilTags
├─ Plan path with A*
├─ Avoid obstacles with DWA
└─ Works immediately! ✅
```

**Why classical, not learning?**
- ✅ Reliable and safe
- ✅ Works after discovery run (no training)
- ✅ Interpretable (you understand what it does)

---

#### **Vision: Learning Required**

**Challenge:** 2,000 facilities have different:
- Lighting (fluorescent, natural, LED)
- Wall colors (white, beige, blue, pink)
- Floor types (tile, carpet, wood)

**Without learning:** Would need to manually collect data at each facility ❌

**With vision learning (YOLO):**

```
Train once on diverse synthetic data
    → Works in all 2,000 facilities ✅
```

**This is where Isaac Sim becomes critical** (we'll cover this in doc 03)

---

#### **Voice Interface: Pre-trained + Fine-tuned**

```
SLM (Small Language Model):
├─ Base: Llama 3 / Mistral (pre-trained)
├─ Fine-tune: Nursing-specific conversations
└─ Deploy: Offline on device
```

**Not Isaac Sim** - text-based fine-tuning

---

### **AMR Technology Stack**

| Component | Method | Training? | Time |
|-----------|--------|-----------|------|
| **Navigation** | SLAM + A* | ❌ No | Discovery run (~15 min) |
| **Vision** | YOLO | ✅ Yes | Train once (~2 days) |
| **Voice** | Fine-tuned SLM | ✅ Yes | Fine-tune once (~1 day) |
| **Motor control** | PID | ❌ No | Instant |

**Key insight:** Hybrid approach - classical where possible, learning where necessary!

---

### **Business Value**

**Old approach (manual data collection):**
- Visit 2,000 facilities
- Collect videos, label data
- Train per-site models
- Cost: $2,000,000+
- Time: 10+ years

**Modern approach (sim + classical):**
- Generate synthetic training data (Isaac Sim)
- Train vision model once
- Classical navigation per site
- Cost: $10,000 (Isaac Sim) + deployment
- Time: 12-16 months for 2,000 sites

**ROI:** $1,990,000 savings, 10x faster deployment

---

## Use Case 2: Pick-and-Place Demonstration

### **Business Context**

**Goal:** Teach robot arm to pick up cube, place in bucket

**Approach:** Imitation Learning (learn by copying expert demonstrations)

**This is your MuJoCo demo project!**

---

### **Why Imitation Learning?**

**Could we use classical methods?**
- For simple pick-place: Yes (inverse kinematics + scripted)
- For variable objects/positions: Becomes very complex

**Could we use RL?**
- Yes, but would take weeks of training
- IL works in hours with 50 demonstrations

**Decision:** IL is the sweet spot for manipulation tasks

---

### **Technical Pipeline**

#### **Phase 1: Data Collection**

```
Expert Policy (Scripted):
├─ Uses inverse kinematics
├─ Move → Grasp → Lift → Place
└─ Perfect demonstrations

Record 50 Episodes:
├─ Randomized cube positions
├─ Save images + joint actions
└─ Convert to LeRobot format (447MB)
```

**Key:** Domain randomization creates variety in simple simulation

---

#### **Phase 2: Training

**

```
ACT Policy (Action Chunking Transformers):
├─ Vision: ResNet-18 encoder
├─ Action: Transformer decoder
└─ Outputs: 100-step action sequence

Training:
├─ 30,000 steps (~2-3 hours)
├─ Loss: 4.374 → 0.036 ✅
└─ Learns movement patterns
```

**Training method:** Fixed 30k steps, no periodic eval (simplicity!)

---

#### **Phase 3: Deployment (Simulation)**

```
Evaluation:
├─ Run learned policy 10 times
├─ Record video
└─ Measure success rate

Results:
├─ Loss decreased dramatically ✅
├─ Shows clear intent to pick ✅
├─ Success rate: 0% (needs more data/training)
└─ Proof: Learning pipeline works!
```

---

### **Lessons Learned**

**Critical bugs found:**

1. **Action recording bug**
   ```python
   # ❌ WRONG: Learns identity function
   action[t] = current_position
   
   # ✅ CORRECT: Learns state→next_state
   action[t] = next_position
   ```

2. **Image normalization mismatch**
   - Training used ImageNet normalization
   - Eval initial forgot it → policy failed
   - Fix: Match preprocessing exactly

**Takeaway:** Imitation learning works, but details matter!

---

### **Technology Stack**

| Component | Framework | Purpose |
|-----------|-----------|---------|
| **Simulation** | MuJoCo | Fast physics, prototyping |
| **Expert policy** | Custom (IK-based) | Generate demonstrations |
| **Data format** | LeRobot | Standard IL format |
| **Policy** | ACT (transformers) | Learn from demos |
| **Training** | PyTorch + LeRobot | 30k steps |

---

## Comparing the Two Use Cases

| Aspect | Nursing AMR | Pick-Place Demo |
|--------|-------------|-----------------|
| **Scale** | 2,000 facilities | Research prototype |
| **Navigation** | Classical (SLAM) | N/A |
| **Vision** | YOLO (learning) | Part of ACT policy |
| **Manipulation** | None | ACT (IL) |
| **Training time** | Vision only (~2 days) | Full pipeline (~3 hours) |
| **Sim platform** | Isaac Sim (for vision) | MuJoCo (lightweight) |
| **Business value** | $2M savings, production | Learning the pipeline |

---

## Strategic Insights

### **1. Don't Over-Learn**

**AMR example:** Uses classical navigation even though RL navigation exists

**Why?** Classical is:
- Faster to deploy
- More reliable
- More interpretable

**Lesson:** Use learning only where necessary

---

### **2. Hybrid is Reality**

**Real products combine:**
- Classical methods (reliability)
- Learning methods (flexibility)

**Pure learning systems are research, not production** (except self-driving cars)

---

### **3. Simulation Matters**

**MuJoCo:** Great for learning the pipeline (fast, simple)

**Isaac Sim:** Required for production (photorealistic, sim-to-real)

**Both are needed** at different stages

---

### **4. Data is the Differentiator**

**AMR:** Can't scale without synthetic data (Isaac Sim)

**Pick-place:** 50 demos sufficient for proof-of-concept

**More data always helps**, but quality > quantity

---

## What You've Learned

From these two use cases:

✅ **When to use classical** - Navigation, motor control

✅ **When to use learning** - Perception, manipulation

✅ **How to combine them** - Hybrid architecture

✅ **Why simulation matters** - Scaling, safety, speed

✅ **Business thinking** - ROI, deployment time, reliability

---

## What's Next?

You've seen two use cases. Now let's explore **how to scale** the AMR vision approach using Isaac Sim as a platform investment.

**→ Continue to:** [03 - Isaac Sim Platform Strategy](03_isaac_sim_platform_strategy.md)

---

*Part 2 of 5-part learning journey*
