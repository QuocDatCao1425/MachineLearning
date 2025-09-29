# DQN on SpaceInvaders

## Videos

**Early Training (untrained / random policy):**  
<video src="early.mp4" controls width="480"></video>  
[Download / View Early Video](early.mp4)

**Later Training (after learning):**  
<video src="later.mp4" controls width="480"></video>  
[Download / View Later Video](later.mp4)

---

## Reflection

**Why SpaceInvaders?**  
I chose *SpaceInvaders* because it combines **accuracy** (aiming at enemies) with **dodging** (avoiding incoming fire). Compared to Pong, rewards are sparser and delayed — you only score when an enemy is hit, and survival also matters. This makes it a richer environment for testing DQN.

**Early vs. Later Behavior**  
In the **early video**, the agent plays almost randomly: it shoots constantly without lining up, misses most targets, and is eliminated quickly. In the **later video**, after training, the agent survives longer, aims shots more effectively, and manages to rack up a noticeably higher score. The difference illustrates the learning progress even after a relatively short training run.

**Challenges and Next Steps**  
The main challenges were unstable training and the tendency to “solve” too quickly with a low reward bound. To improve performance I would:
- Increase replay buffer size to 1e6 and warmup to 50k steps.  
- Use slower epsilon decay (over 1e6 frames, final epsilon ~0.05).  
- Train for longer (≥500k frames) to stabilize behavior.  
- Try improvements such as **Double DQN**, **Dueling networks**, and **prioritized replay**.  

---

## Files in this Repo
- `early.mp4` – gameplay before training (random).  
- `later.mp4` – gameplay after training (improved).  
