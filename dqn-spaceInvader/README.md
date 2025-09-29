
# DQN on SpaceInvaders

## Videos

**Early Training (untrained / random policy):**  
<video src="early.mp4" controls width="480"></video>  

**Later Training (after learning):**  
<video src="later.mp4" controls width="480"></video>  

---

## Reflection

**Why SpaceInvaders?**  
I chose SpaceInvaders because it combines both **accuracy** (aiming and shooting) and **dodging** (avoiding incoming fire). Compared to Pong, it has **sparse and delayed rewards**, making it a harder test for DQN.

**Early vs. Later Behavior**  
- In the **early video**, the agent fires randomly, rarely hits enemies, and gets eliminated quickly.  
- In the **later video**, after some training, the agent survives longer, lines up shots better, and achieves a higher score.  

**Challenges and Next Steps**  
- Training can be unstable with a small replay buffer and fast epsilon decay.  
- To improve further I would:  
  - Increase replay buffer to 1e6 and use slower epsilon decay (over ~1M frames).  
  - Train longer (≥500k frames).  
  - Use **Double DQN**, **Dueling networks**, and **prioritized replay**.  

---

## Files in this Repo
- `early.mp4` – gameplay before training.  
- `later.mp4` – gameplay after training.  

