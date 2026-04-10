# Variational Quantum Eigensolver (VQE) for $H_2$ Molecular Ground State Simulation

This repository contains a comprehensive, physics-informed implementation of a Variational Quantum Eigensolver (VQE) used to calculate the ground-state energy of a Hydrogen ($H_2$) molecule. 

Built using **PennyLane, JAX, and Optax**, this project serves as a practical exploration of Quantum Chemistry simulations. Rather than treating the quantum machine learning (QML) libraries as black boxes, this notebook explicitly breaks down the physical symmetries, Hamiltonian mappings, and gradient-descent mathematics required to simulate molecular systems on quantum hardware.

---

## 🔬 What I Have Done (Key Contributions & Insights)

As a researcher transitioning from Theoretical Physics to Quantum Computing, my main objective with this project was to explicitly connect the underlying quantum mechanical principles to the software implementation. 

If you are reviewing this code, here is exactly what I have designed, implemented, and explained within the notebook:

### 1. Physics-Informed Ansatz Preparation
Instead of using a brute-force hardware-efficient ansatz, I constructed a chemically inspired, parameterized trial wavefunction: 
$|\Psi(\theta)\rangle = \cos(\theta/2)\,|1100\rangle - \sin(\theta/2)\,|0011\rangle$

I explicitly break down the physical reasoning behind this specific state preparation:
* **Qubit Mapping:** Explained the Jordan-Wigner (JW) transformation, justifying why a minimal basis set (STO-3G) requires exactly 4 qubits for the $H_2$ molecule (2 spatial orbitals $\times$ 2 spin states).
* **Spin Symmetry constraints:** Derived why the optimization must focus on the **singlet state** ($m_s = 0$, antisymmetric spin wavefunction) to find the true ground state, inherently excluding higher-energy triplet states ($|\uparrow \uparrow\rangle, |\downarrow \downarrow\rangle$).
* **Spatial Symmetry constraints:** Mathematically justified the exclusion of intermediate states like $|0110\rangle$ and $|1001\rangle$ which break the spatial symmetry of the molecule.

### 2. Hamiltonian Construction
* Extracted the molecular Hamiltonian from the `qchem` database.
* Demonstrated a secondary method to compute the Hamiltonian directly from fundamental atomic coordinates and bond lengths ($0.701 \mathring{A}$).
* Printed and analyzed the 16x16 sparse Hamiltonian matrix to verify the theoretical mapping.

### 3. Custom JAX/Optax Optimization Loop
To demonstrate a clear understanding of the optimization mathematics, I bypassed standard built-in QML optimizers and built a custom training loop:
* Utilized **JAX** for just-in-time compilation (JIT) and high-precision (`float64`) automatic differentiation (`jax.grad`).
* Implemented Stochastic Gradient Descent (SGD) via **Optax**, manually applying the gradient updates to the circuit parameter $\theta$ over 100 iterations.
* Configured a convergence tolerance mechanism ($\Delta E \leq 10^{-6}$) to efficiently halt the simulation once the ground state was achieved.

### 4. Evaluation vs. Exact Limits
* Benchmarked the VQE calculated energy ($E_{HF}$) against the exact classical Full Configuration Interaction ($E_{FCI}$) energy.
* Plotted the energy convergence and gate parameter ($\theta$) evolution using `matplotlib`.
* Converted the final optimized $\theta$ back into the explicit quantum state vector.

---

## 🛠️ Technology Stack
* **Quantum Framework:** PennyLane
* **Optimization & Autodiff:** JAX, Optax
* **Visualization:** Matplotlib
* **Language:** Python 3 (Jupyter Notebook)

---

## 🚀 Running the Notebook

**1. Install the required dependencies:**
```bash
pip install jax pennylane optax matplotlib
```
*Note: JAX is configured to run on the CPU with 64-bit precision to ensure chemical accuracy during expectation value calculations.*

**2. Run the Notebook:**
Simply execute the cells in `VQE-Molecular-GroundState-Sim.ipynb` sequentially. The code will output the Hamiltonian, the iterative step-by-step energy reduction (measured in Hartree), and the final convergence plots.

---

## 💡 Future Research Directions

At the end of the notebook, I have outlined specific "Load Bearing" and "Tangential" research questions that I aim to explore further, including:
1. Identifying generalized patterns for constructing Hartree-Fock states for complex molecules (3+ electrons).
2. Deepening the mathematical formulation and cost-analysis of using different classical optimizers (e.g., Adam vs. L-BFGS vs. SPSA) within noisy quantum environments.
3. Scaling these simulations to execute on real superconducting quantum hardware via Qiskit/Cirq APIs to study the impact of hardware noise on spatial/spin symmetries.

---

## 📚 Resources & References

This implementation and the foundational quantum chemistry concepts explored within the notebook were adapted from the official PennyLane documentation:

* **Primary Reference:** https://pennylane.ai/qml/demos/tutorial_vqe

---
*Developed by Rithik Rai.*
