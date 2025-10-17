# MNIST GAN: Conceptual Questions for Engineering Students

## **Question 1: Adversarial Balance** �
Looking at your GAN training:
```python
# Generator wants discriminator to output 1.0 (real)
# Discriminator wants to output 0.0 for fake images
```
**What happens if the Discriminator becomes too good at detecting fakes? How would you fix this problem during training?**

---

## **Question 2: Architecture Power Balance** <�
Your GAN has:
- Generator: 559,632 parameters  
- Discriminator: 533,505 parameters

**If you made the Generator 10x more complex but kept the Discriminator simple, what training problems would occur and why?**

---

## **Question 3: Loss Pattern Analysis** =�
During training you see:
```
Epoch 10: Gen Loss = 0.72, Disc Loss = 0.67
Epoch 20: Gen Loss = 1.60, Disc Loss = 0.16
```
**Is this good or bad training? Explain what's happening between these two epochs.**

---

## **Question 4: Scaling to Complex Data** =,
Your code generates 28�28 MNIST digits (784 pixels). For 512�512 medical images (262,144 pixels):
```python
nn.Linear(512, 28 * 28)      # Current: 14,336 parameters
nn.Linear(512, 512 * 512)    # Medical: 134 million parameters!
```
**Why won't this fully-connected approach work for high-resolution images? What architectural change is needed?**

## **Question 5: Generator Shortcut Exploitation 🎯**

Scenario: During MNIST GAN training, your generator discovers that creating images with a specific noise pattern (like static/dots) in the corners consistently fools your discriminator into thinking they're "real" - even though they don't look like actual digits at all.

Your training logs show:
Epoch 100: Gen Loss  = 0.1 (very low - generator is "winning")
Epoch 100: Disc Loss = 0.8 (high - discriminator is confused)

But when you look at generated images, they're just meaningless static that somehow tricks the discriminator. Is this possible?