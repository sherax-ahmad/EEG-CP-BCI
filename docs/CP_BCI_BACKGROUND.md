# Clinical Background: EEG-BCI for Cerebral Palsy

## What is Cerebral Palsy (CP)?

Cerebral Palsy is a group of permanent movement and posture disorders caused by non-progressive brain lesions occurring before, during, or shortly after birth. It affects approximately **1 in 500 live births** globally.

**Key distinction:** In most forms of CP, especially spastic types, the **motor cortex and intention pathways are intact**. The failure occurs in the **corticospinal tract** — the pathway from motor cortex to spinal cord and muscles. Children with CP *intend* to move; the signal simply fails to reach the muscles properly.

This is exactly why BCI technology is so promising for CP: we can read the **intact intention signal** from EEG, even when the downstream motor pathway is compromised.

---

## Neurophysiology Relevant to BCI

### Motor Imagery in Healthy Subjects
During imagined movement (motor imagery, MI), the brain produces EEG changes nearly identical to actual movement:
- **Mu rhythm ERD (8–12 Hz):** Power decrease over contralateral motor cortex
- **Beta ERD (13–30 Hz):** Broadband suppression during active motor planning
- **Beta ERS (rebound, ~20–30 Hz):** Post-movement power increase, indicating movement cessation

### Motor Imagery in CP
Research shows that **many children with CP retain motor imagery ability**, though potentially altered:
- Souto et al. (2020, Dev Med Child Neurol): Children with unilateral CP show MI capacity in case-control study
- Steenbergen et al. (2009): MI possible in CP, especially for actions the child can still physically perform
- The mu/beta ERD pattern is present but may be:
  - Reduced in amplitude
  - More bilateral (less lateralized) due to compensatory reorganization
  - More variable across individuals

### Key EEG Findings in CP
- More diffuse EEG patterns vs. healthy controls
- Increased theta (4–8 Hz) power, especially in spastic forms
- Intact P300 responses (for passive BCI paradigms)
- Variable mu/beta ERD depending on lesion location and severity

---

## Which Children with CP Are Best Candidates?

Based on published literature:

| Criterion | Better Outcome | Worse Outcome |
|-----------|---------------|---------------|
| CP type | Spastic diplegia, hemiplegia | Dyskinetic, ataxic |
| Lesion | Periventricular leukomalacia | Extensive cortical damage |
| Age | 6+ (better cognitive engagement) | Under 4 (reliability issues) |
| Cognition | Age-appropriate | Significant ID |
| MI ability | Confirmed via imagery scales | Unable to imagine movements |

---

## BCI Paradigms for CP

### 1. Motor Imagery BCI (this project)
- **Signal:** Imagined L/R hand movement → CSP → classifier
- **Best for:** Children who can perform mental imagery
- **Advantage:** Most intuitive; exploits natural motor planning
- **Challenge:** Requires cognitive engagement; training period needed

### 2. P300 Speller BCI
- **Signal:** P300 ERP elicited by target flashes
- **Best for:** Communication (matrix speller)
- **Advantage:** Works without motor imagery ability
- **Challenge:** Requires visual attention; slower communication

### 3. SSVEP BCI
- **Signal:** Steady-state visual evoked potential
- **Best for:** Gaze-dependent control
- **Advantage:** High accuracy, minimal training
- **Challenge:** Requires stable gaze control (difficult in some CP)

---

## Dataset Note for Researchers

**No large open EEG dataset exists specifically for children with CP performing motor imagery.**

The datasets used in this project (PhysioNet, BCI Competition IV) are from **healthy adult subjects**. This is acceptable for:
- Algorithm development and benchmarking
- Pipeline validation
- Proof-of-concept

For **actual CP deployment**, you would need:
1. Ethics approval from institutional review board
2. Collaboration with a pediatric neurology/rehab center
3. Age-appropriate protocols (shorter sessions, gamified feedback)
4. Consent from parents/guardians + assent from child

**Relevant existing CP-BCI papers:**
- Daly et al. (2013). "On the control of brain-computer interfaces by users with cerebral palsy." *Clin Neurophysiol*, 124(9):1787-97.
- Behboodi et al. (2024). "Development and evaluation of a BCI-neurofeedback system with real-time EEG detection for neurorehabilitation of children with cerebral palsy." *Front Hum Neurosci*, 18:1346050.
- Xie et al. (2021). "Rehabilitation of motor function in children with CP based on motor imagery." *Cogn Neurodyn*, 15:939-948.
- Qu et al. (2024). "Exploring cortical excitability in CP children through lower limb robot training based on MI-BCI." *Sci Rep*.

---

## Ethical Considerations

1. **Informed consent:** Full parental consent + child assent (age-appropriate)
2. **Data privacy:** EEG data is sensitive health data; store encrypted, anonymized
3. **Do no harm:** BCI training should not replace, but complement, physiotherapy
4. **Realistic expectations:** Communicate limitations; BCI is a tool, not a cure
5. **Accessibility:** Any deployed device must be affordable and culturally appropriate

---

## Glossary

| Term | Definition |
|------|-----------|
| BCI | Brain-Computer Interface |
| CSP | Common Spatial Patterns — spatial filter for MI-EEG |
| CP | Cerebral Palsy |
| ERD | Event-Related Desynchronization — power decrease |
| ERS | Event-Related Synchronization — power increase |
| ICA | Independent Component Analysis |
| LSL | Lab Streaming Layer — real-time data streaming protocol |
| MI | Motor Imagery — imagined movement |
| GMFCS | Gross Motor Function Classification System (I–V) |
| Mu rhythm | 8–12 Hz sensorimotor oscillation |
| Beta rhythm | 13–30 Hz, strongly linked to motor states |
