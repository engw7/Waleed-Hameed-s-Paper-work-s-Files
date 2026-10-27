# Isolating the Contribution of Deep Neural Networks in Hybrid Scan-Based MPPT Controllers Under Partial Shading

Waleed Hameed¹,²\*, Kamil Dimililer³,⁴

¹\*Department of Electrical Engineering Technologies, Al-Huda University College, ZIP: 31001, Ramadi, Al Anbar, Iraq

²\*Scientific Affairs Division, Al-Huda University College, ZIP: 31001, Ramadi, Al Anbar, Iraq

³ Department of Electrical and Electronic Engineering, Near East University, Near East Boulevard, ZIP: 99138, Nicosia, North Cyprus, via Mersin 10, TURKEY

⁴ Applied Artificial Intelligence Research Centre (AAIRC), Centre for Technology and Engineering (BILTEM), Near East University, Near East Boulevard, ZIP: 99138, Nicosia, North Cyprus, via Mersin 10, TURKEY

\*Corresponding author(s). E-mail(s): waleed.hameed@uoalhuda.edu.iq Contributing authors: kamil.dimililer@neu.edu.tr

## Abstract

Maximum power point tracking (MPPT) under partial shading is often addressed with neural networks, but their contribution relative to simpler global-search strategies is rarely isolated. This study proposes a two-stage hybrid controller in which a 14-point I–V scan locates the global-peak region and perturb and observe (P&O) refines it, and compares a deep neural network (DNN) scan stage with a non-ML best-probe rule and six other trackers, using a single-diode model and 16 years of PVGIS-SARAH3 data. The hybrid resolved all nine shading patterns at 99.3% efficiency against 78–80% for P&O, but Scan+P&O without a network performed equally well (99.36% versus 99.32%, p = 0.23), so the gain stems from the scan. The DNN gave faster cold-start convergence (1.0 versus 4.0 cycles) but was outperformed outside its training range. Its computational burden lies in the offline, parallelizable data generation, not in deployment.

**Keywords:** maximum power point tracking; partial shading; deep neural network; hybrid controller; domain randomization; high-performance computing

## Introduction

A photovoltaic (PV) module does not deliver its power at a fixed operating point. Its current–voltage (I–V) characteristic is nonlinear and shifts continuously with irradiance and cell temperature [1], so the voltage at which the module yields maximum power changes throughout the day. Maximum power point tracking (MPPT) algorithms address this by adjusting the duty cycle of a DC–DC converter so that the module stays at its maximum power point (MPP), and the quality of the tracker therefore directly determines how much of the available energy is harvested [2, 3].

The most widely used trackers are Perturb and Observe (P&O) and Incremental Conductance (IncCond). Both require little computation and are established in stand-alone and grid-connected systems [2, 4]. P&O perturbs the operating point and keeps or reverses the direction according to the observed change in power [5, 6]. IncCond compares incremental and instantaneous conductance to decide on which side of the MPP the system operates [3, 7]. Their limitations are well documented: steady-state oscillation around the MPP, which wastes energy [8], and slow or unreliable response when irradiance and temperature change quickly [9–11].

A more serious limitation appears under partial shading. When the modules of a series string receive different irradiance, bypass diodes conduct and the P–V curve develops several local maxima. A hill-climbing method that settles near a local maximum cannot know that a higher global maximum exists elsewhere on the curve, and it may stay there indefinitely (Fig. 1). The simulations of the present study show this directly: P&O reached the global maximum in only three of nine shading patterns and harvested 80.4% of the available energy on average. This behaviour is well known: conventional trackers can lock onto a local maximum under partial shading [4].

&#91;image: Fig. 1\]

**Fig. 1** Effect of partial shading on a series string of three PV modules with bypass diodes (illustrative single-diode simulation, 1000/600/300 W/m² against uniform 1000 W/m²): (a) the P–V curve acquires three maxima, of which only one is global, and a hill-climbing tracker that settles at a local maximum loses a large share of the available power; (b) the corresponding I–V curve shows the current steps produced by the bypass diodes.

Artificial neural networks (ANNs) have been proposed to overcome these limitations. They learn the nonlinear mapping from measurable quantities, such as irradiance and temperature, to the operating point, so they can respond without perturbation and avoid steady-state oscillation [12, 13]. A network is a layered arrangement of weighted neurons whose weights are adjusted during training so that its output approximates the desired control signal [14, 15]. ANN-based controllers have been reported to improve convergence speed and tracking accuracy [8, 16, 17].

The reported gains, however, are difficult to interpret. A network is usually evaluated against P&O or IncCond as a complete controller, so any improvement is attributed to the ANN even when it may come from other design choices, such as an explicit search over the P–V curve. Hybrid designs that combine a network with a perturbation stage have appeared recently [1, 13, 18, 19]; in the last of these, the network supplies a starting estimate of the global maximum under partial shading and an adaptive P&O routine refines it. To the authors’ knowledge, such designs are compared against P&O alone or against a metaheuristic, and not against the same hybrid structure without the network, so the network’s own contribution is not isolated. In addition, many studies validate on a limited set of scenarios, omit the cost of the search or scan stage, and do not test the network on plants whose parameters differ from those used in training. It therefore remains unclear how much of the benefit of a hybrid ANN controller is due to the network and how much to the scan-and-track structure around it.

This study addresses that question. It develops a two-stage hybrid controller in which a fast I–V scan gives a coarse description of the P–V curve, a deep neural network (DNN) converts it into an estimate of the global-maximum voltage, and P&O then refines the estimate (Fig. 2). The controller is compared with a scan-based controller of identical structure but without a neural network, with a DNN-only variant, and with P&O, IncCond, a polynomial regression, a conventional ANN-MPPT baseline, a particle-swarm global tracker and a variant in which a gradient-boosted-tree regressor replaces the DNN. A boost-converter plant, a design ablation, sensor-noise and quantization tests and multi-seed statistics test how far the conclusions hold. Because the controllers differ in one component at a time, the contribution of the DNN can be separated from that of the scan strategy. The evaluation uses a corrected five-parameter single-diode model driven by 16 years of hourly PVGIS-SARAH3 irradiance and temperature for Ramadi, Iraq [20].

&#91;image: Fig. 2\]

**Fig. 2** Overview of the proposed two-stage hybrid controller: a 14-point I–V scan feeds the DNN, which estimates the global-maximum voltage with a conformal interval; P&O then refines the operating point and drives the boost converter. A re-scan is triggered when P/G falls below 90% of its running maximum or every 300 cycles. In the non-ML variant the DNN is replaced by the best-power probe voltage.

The topic is relevant to high-performance and real-time computing for two reasons. First, a shading-robust DNN must be trained over a very large family of plant configurations and shading patterns; this study generates 200,000 domain-randomized plants, each requiring a full I–V solution, which is a parallel simulation workload that grows quickly with the number of parameters, and a credible evaluation (multi-seed statistics, design ablations, noise and converter studies) multiplies the number of independent simulation runs further. Second, an MPPT controller runs on embedded hardware under strict per-cycle time limits, so the computational cost of inference is itself a design criterion. The results quantify both aspects: parallel data generation gave a measured speed-up of 2.62× on four workers (1.62× on two) and scenario-seed sweeps 3.82× on four workers, while the hybrid controller needs about 8.9 µs per control cycle and a network inference about 19 µs, run only at scan instants, a negligible share of the 1 s control period. Parallel resources therefore matter for the offline pipeline and not for the online controller, and this study measures where each applies.

The main contributions are as follows:

1. A hybrid scan–DNN–P&O controller that reaches the global maximum in all nine evaluated shading patterns, with 99.3% tracking efficiency compared with 78–80% for P&O and a conventional ANN-MPPT.
2. A controlled ablation that separates the DNN from the scan strategy. A scan-based controller without a neural network matches the hybrid under shading (99.36% versus 99.32%), and outside the training range a best-probe rule outperforms the DNN (99.37% versus 98.44%); a gradient-boosted-tree estimator and networks with 1,633 to 103,169 parameters give the same closed-loop efficiency, and the measurable benefits of the DNN are faster convergence after a scan and a more accurate first estimate.
3. A shading-aware, domain-randomized training pipeline with conformal prediction intervals, which attain 0.899 coverage at the 0.90 nominal level within the training distribution and expose a drop to 0.41 for plants outside it.
4. A reproducible evaluation that includes scan overhead in the efficiency, reports plant-mismatch sweeps, cold-start behaviour, a boost-converter plant, sensor-noise and quantization tests, paired multi-seed statistics and a measured analysis of where parallel computing and real-time inference matter, and states the cases in which the hybrid does not outperform P&O.

## Related Work

This section reviews the main families of MPPT methods, states the strengths and weaknesses of each, and compares representative studies with the present work in Table 1.

### Conventional methods

P&O and IncCond remain the reference trackers because they need only voltage and current measurements, are cheap to implement and have been standardized in commercial converters [2, 3]. Their strengths are simplicity, low computation and independence from the PV model. Their weaknesses are a trade-off between speed and steady-state oscillation set by the perturbation step, slow response to rapid irradiance changes, and, above all, the inability to distinguish a local from a global maximum [5, 6]. Improved IncCond variants, for example with an integral regulator, reduce oscillation around the peak but remain single-peak methods [7]. Comparative studies under partial shading confirm that conventional trackers can converge to a local maximum [4].

### Global-search and learning methods for partial shading

Two strategies address multi-peak curves. Metaheuristic searches, such as particle swarm optimization combined with sliding-mode extremum seeking [11] or the wolf pack algorithm [9], explore the curve and can find the global maximum, but they need many iterations, several tuning parameters and a convergence phase during which power is lost. Learning-based controllers move this cost offline. Reinforcement learning has been applied to MPPT with deep Q-networks, which reduce the number of tracking steps relative to P&O and IncCond [21], and with a fuzzy–DDPG controller that reached 95% tracking efficiency under partial shading compared with 85% for P&O and 88% for IncCond [22]. These controllers need extensive training before deployment. Review articles agree that advanced methods deliver higher efficiency than conventional ones at the price of greater design complexity and cost [2, 3].

### ANN-based MPPT

ANN-MPPT learns the mapping from irradiance and temperature, and sometimes voltage and current, to the optimal operating point. Reviews report average efficiencies of about 98% under uniform conditions together with faster convergence and fewer oscillations [12], and a simulation benchmark ranked the ANN highest at 98.6% [3]. An experimentally validated data-driven ANN reached 98.16% efficiency with a tracking time of 1.3 s [8]. Design choices studied in the literature include alternative network inputs and genetic-algorithm selection of the topology [15], comparison with support vector machines trained on P&O-generated data [14], and the choice of DC–DC converter [16, 17]. A descriptive simulation study reports that ANN-MPPT outperforms P&O in efficiency and accuracy under dynamic weather and shading [10]. The weaknesses are shared: a network trained on a model of one plant carries that model’s errors into the controller, results are mostly obtained in simulation from the same model that generated the training labels, partial shading is often handled by adding inputs rather than by an explicit search, and the computation per control cycle is rarely reported.

### Hybrid ANN and P&O methods

Hybrids use the network to initialize or to tune a perturbation-based tracker. A P&O–ANN hybrid has been compared with a P&O–PSO hybrid under uniform and non-uniform irradiance, with better power extraction and tracking speed [13]. A network has been used to select the scaling factor of an adaptive P&O step under partial shading [1], or to estimate the duty-cycle change after sudden irradiance changes [18]. The closest work to the present study uses a small multilayer perceptron to estimate the global maximum power point under partial shading and then refines it with an adaptive P&O routine, improving efficiency by 15.6–49% over P&O alone in simulation and matching a differential-evolution benchmark with less computation [19]. Comprehensive comparisons also list the hybrid neural-network-plus-P&O method among the strongest options for dynamic conditions [6].

### Comparison and research gap

Table 1 summarizes representative studies. Two features distinguish the present work. First, the hybrid is compared with the same architecture without a network, so the contribution of the DNN is isolated from that of the scan strategy; none of the studies above reports such a control. Second, the evaluation covers properties that the studies above rarely report together: global-maximum success on nine shading patterns, energy lost to scanning, robustness to plant mismatch, behavior outside the training range with calibrated uncertainty, and computation per control cycle.

### Critical comparison with the closest ANN studies

Two of the studies in Table 1 were read in full and are compared with the present work in detail. Abouzeid et al. [8] train two networks, one taking irradiance and temperature and one taking current and temperature, that output the duty cycle directly. The networks were trained on about 6,100 samples and validated on an Arduino-based prototype against P&O and IncCond, with experimental efficiencies of 97.99–98.25%. The evaluation uses uniform irradiance (steps between about 812 and 398 W/m²), so the multi-peak problem is not addressed, the network is the only decision-maker, and no variant without the network is reported. The authors also list limited controller resources and the cost of an irradiance sensor as limitations. The present study differs in three ways: it targets partial shading, it uses a scan to measure the curve rather than an irradiance sensor, and it reports what the network adds on top of the scan. Sørensen and Mihet-Popa [4] compare P&O, IncCond, fuzzy logic, grey wolf optimization, PSO and two ANNs on three series modules in five Simulink scenarios (one uniform, four shaded) at a fixed 25 °C. Their results show that the benefit of an ANN depends on its inputs: the network trained on meteorological data fell to 35.75–96.60% under shading, whereas the network fed with module irradiances stayed near 99%. This agrees with the finding here that the information supplied to the network, not its depth, determines performance. Their study is simulation only, uses fixed temperature, and does not include a hybrid in which a perturbation stage corrects the network estimate. The studies by Bouchetob and Nadji [16], Gugulothu and Naik [13] and Shyam and Ghosh [10] were not available to the authors in full text; they are therefore described in this section only at the level of their published abstracts, and no quantitative comparison with them is made. No cited study re-implements a competing published controller under identical conditions, and neither does the present work beyond the standard algorithms and PSO (Limitation 8).

**Table 1** Representative MPPT studies and their relation to the present work. Results are those reported by the authors in the cited abstracts; “Not reported” and “not described” mean that the abstract does not state it.

| Study | Method | Partial shading | Reported result | Gap relative to this work |
| --- | --- | --- | --- | --- |
| Abouzeid et al. [8] | Data-driven ANN, simulation and experiment | No (uniform irradiance steps) | 97.99–98.25% experimental efficiency; simulated convergence 0.03 s versus 0.08–0.4 s for P&O/IncCond | No shading; the network alone sets the duty cycle; no hybrid or ablation |
| Shyam & Ghosh [10] | ANN versus P&O (Simulink) | Yes | ANN better in efficiency and accuracy | Simulation only; no ablation of the ANN |
| Nour Ali [15] | ANN with alternative inputs, GA-designed topology | Not reported | Improved ANN performance in simulation | Shading not described |
| Pradeep Kumar et al. [14] | ANN and SVM trained on P&O data | Not reported | MSE 0.5 (ANN), 0.159 (cubic SVM) | Prediction error only; no closed-loop efficiency |
| Sørensen & Mihet-Popa [4] | P&O, IncCond, FLC, GWO, PSO, ANN compared (Simulink) | Yes (four shaded scenarios, fixed 25 °C) | Irradiance-input ANN about 99% under shading; weather-input ANN 35.75–96.60% | Simulation only; no hybrid structure and no ablation |
| Gugulothu & Naik [13] | P&O–ANN hybrid versus P&O–PSO | Yes (non-uniform) | Better extraction and speed than P&O–PSO | Not compared with the hybrid without ANN |
| Krishnaram et al. [1] | ANN-adaptive step-scaled P&O | Yes | Performance enhancement reported | ANN tunes a step size, not the global search |
| Dawahdeh et al. [18] | NN with P&O for sudden irradiance changes | Not reported | Better than P&O under sudden changes | Shading not described; no ablation |
| Cruz et al. [19] | MLP estimate of GMPP, adaptive P&O | Yes | +15.6–49% efficiency versus P&O alone | Simulation only; compared with P&O and DE, not with a no-network hybrid |
| Ortiz-Muñoz et al. [22] | Fuzzy–DDPG reinforcement learning | Yes | 95% versus 85% (P&O) and 88% (IncCond) | Simulation; extensive training; no hybrid ablation |
| **This work** | Scan + DNN + P&O, with no-network and DNN-only ablations | Yes, nine patterns | 99.3% under shading versus 80.4% (P&O); no-network scan matches the hybrid | Simulation only; scan assumed to be one control cycle; DNN advantage limited to cold-start precision |

## Methodology

The methodology follows the order in which the controller is built and evaluated: a physics-based PV model that generates both the training labels and the closed-loop plant; the real irradiance and temperature data that drive it; the two-stage hybrid controller; the deep neural network (DNN) that serves as its global-search estimator; split-conformal uncertainty quantification; the baseline controllers and the evaluation protocol; and the performance metrics. Equations are numbered consecutively and symbols are defined where they first appear.

### PV cell and module model

Each cell is modeled with the five-parameter single-diode equivalent circuit (Fig. 3a), following De Soto et al. [23] and Duffie & Beckman [24]. For a module of N\_s = 54 series-connected cells, the terminal current I at terminal voltage V satisfies

```latex
I = I_{ph} - I_0\left[\exp\left(\frac{V + I R_s}{a}\right) - 1\right] - \frac{V + I R_s}{R_{sh}} \qquad\qquad (1)
```

where I\_ph is the light-generated current, I\_0 the diode saturation current, R\_s and R\_sh the series and shunt resistances, and a the modified ideality factor,

```latex
a = \frac{n\,N_s\,k\,T_k}{q} \qquad\qquad (2)
```

with n the diode ideality factor, k the Boltzmann constant, q the elementary charge and T\_k = T\_c + 273.15 the cell temperature in kelvin. The dependence of the parameters on irradiance G (W/m²) and cell temperature follows De Soto et al. [23]:

```latex
I_{ph} = \frac{G}{G_{ref}}\left[I_{ph,ref} + \mu_{I_{sc}}\,(T_k - T_{ref})\right] \qquad\qquad (3)
```

```latex
I_0 = I_{0,ref}\left(\frac{T_k}{T_{ref}}\right)^{3}\exp\left[\frac{E_g}{n\,k}\left(\frac{1}{T_{ref}} - \frac{1}{T_k}\right)\right] \qquad\qquad (4)
```

```latex
R_{sh} = R_{sh,ref}\,\frac{G_{ref}}{G} \qquad\qquad (5)
```

with reference conditions G\_ref = 1000 W/m² and T\_ref = 298.15 K, a silicon band gap E\_g = 1.121 eV and a short-circuit-current temperature coefficient μ\_Isc = 0.06 %/K of I\_sc. The series resistance R\_s and the ideality factor n do not depend on G. The cell temperature follows from the ambient temperature through the standard NOCT relation, with NOCT = 45 °C:

```latex
T_c = T_{amb} + \frac{NOCT - 20}{800}\,G \qquad\qquad (6)
```

Because Eq. (1) is implicit in I, it is solved in closed form with the Lambert-W function W(·) [25],

```latex
I = \frac{R_{sh}(I_{ph} + I_0) - V}{R_s + R_{sh}} - \frac{a}{R_s}\,W\!\left\{\frac{R_s R_{sh} I_0}{a\,(R_s + R_{sh})}\exp\left[\frac{R_{sh}\,\big(R_s (I_{ph} + I_0) + V\big)}{a\,(R_s + R_{sh})}\right]\right\} \qquad\qquad (7)
```

evaluated with a numerically stable routine for large arguments, so that no iterative root-finding is needed at any simulation step. Figure 3a shows the equivalent circuit.

&#91;image: Fig. 3\]

**Fig. 3** Photovoltaic model. (a) Five-parameter single-diode equivalent circuit (photocurrent source I\_ph, diode D, shunt resistance R\_sh, series resistance R\_s). (b) Series string of three substrings of 18 cells, each protected by a bypass diode; unequal irradiance G₁, G₂ and G₃ produces a multi-peak P–V curve.

**Parameter extraction.** The manufacturer nameplate values supplied with the original dataset were internally inconsistent: the stated I\_mpp exceeded I\_sc, and the stated V\_oc temperature coefficient was incompatible with any physically valid ideality factor at the stated cell count. A consistent parameter set was therefore re-derived. The ideality factor n = 1.2 and the shunt resistance R\_sh,ref = 400 Ω (typical values for silicon) were fixed. For a trial series resistance R\_s, the saturation and light-generated currents follow in closed form from the short-circuit and open-circuit conditions,

```latex
I_{0,ref} = \frac{I_{sc}\,(1 + R_s/R_{sh}) - V_{oc}/R_{sh}}{\left[\exp(V_{oc}/a) - 1\right] - \left[\exp(I_{sc}R_s/a) - 1\right]} \qquad\qquad (8)
```

```latex
I_{ph,ref} = I_{0,ref}\left[\exp(V_{oc}/a) - 1\right] + \frac{V_{oc}}{R_{sh}} \qquad\qquad (9)
```

and R\_s was then found with Brent's root-finding method so that the true maximum power of the model equals the rated power of the datasheet,

```latex
\max_{V}\; V\,I(V;\,R_s) = P_{max} = 244.6\ \mathrm{W} \qquad\qquad (10)
```

The resulting parameters are listed in Table 2. A string of 54 cells was adopted (consistent with the stated 32.9 V open-circuit voltage at about 0.61 V per cell) in place of the 60 cells given in the dataset. The model reproduces the standard-test-condition values exactly (P\_max = 244.6 W, V\_oc = 32.9 V, I\_sc = 9.87 A), and the voltage at the maximum power point (26.6 V) and the temperature coefficients (−0.33 %/K for V\_oc and −0.41 %/K for P\_max) are outputs of the model rather than fitted quantities.

| Quantity | Symbol | Value | Basis |
| --- | --- | --- | --- |
| Short-circuit current | I\_sc | 9.87 A | Datasheet, STC |
| Open-circuit voltage | V\_oc | 32.9 V | Datasheet, STC |
| Maximum power | P\_max | 244.6 W | Datasheet, STC; matched exactly by Eq. (10) |
| Series-connected cells | N\_s | 54 | Consistent with V\_oc at about 0.61 V per cell |
| Diode ideality factor | n | 1.2 | Fixed (typical silicon) |
| Shunt resistance at STC | R\_sh,ref | 400 Ω | Fixed (typical silicon) |
| Series resistance | R\_s | 0.182 Ω | Solved from Eq. (10) |
| Short-circuit-current coefficient | μ\_Isc | +0.06 %/K | Assumed (typical silicon) |
| Nominal operating cell temperature | NOCT | 45 °C | Assumed (standard value) |
| Voltage at maximum power | V\_mpp | 26.6 V | Model output |
| V\_oc temperature coefficient | – | −0.33 %/K | Model output, not fitted |
| P\_max temperature coefficient | – | −0.41 %/K | Model output, not fitted |

**Table 2** Parameters of the single-diode module model at standard test conditions (STC: 1000 W/m², 25 °C cell temperature). Quantities marked as model outputs were not used in the fit.

Unlike a simplified closed-form approximation, this model gives V\_mpp a dependence on irradiance as well as on temperature: at a fixed cell temperature, V\_mpp falls by 2.1%, 5.0% and 8.6% as irradiance drops from 1000 to 400, 200 and 100 W/m², the weak logarithmic shift expected from the diode equation, in addition to the larger and better-known temperature dependence.

**Partial shading.** A series string of three substrings of 18 cells, each with a bypass diode (Fig. 3b), is modeled explicitly, following Patel & Agarwal [26]. Within substring k, n\_k cells receive a reduced irradiance f\_k G (0.10 ≤ f\_k ≤ 0.85) and the remaining 18 − n\_k cells receive G. A group of N\_g cells that has irradiance G\_g, cell temperature T\_g and carries the current I has the terminal voltage obtained by inverting the single-diode equation with the Lambert-W function,

```latex
V_g(I) = R_{sh,g}\,(I_{ph,g} + I_{0,g}) - I\,(R_{s,g} + R_{sh,g}) - a_g\,W\!\left\{\frac{I_{0,g}R_{sh,g}}{a_g}\exp\left[\frac{R_{sh,g}\,(I_{ph,g} + I_{0,g} - I)}{a_g}\right]\right\} \qquad\qquad (11)
```

where I\_ph,g and I\_0,g follow from Eqs. (3) and (4) at (G\_g, T\_g), and a\_g, R\_s,g and R\_sh,g are the quantities of Eqs. (2) and (5) scaled by N\_g/N\_s. The voltage of substring k is the sum of its unshaded and shaded groups, limited from below by the bypass diode, which conducts when the substring would otherwise become negative by more than its forward drop V\_bp = 0.7 V,

```latex
V_k(I) = \max\left\{V_{k,u}(I) + V_{k,s}(I),\; -V_{bp}\right\},\qquad k = 1,2,3 \qquad\qquad (12)
```

and the voltage of the string at the common current I is

```latex
V(I) = \sum_{k=1}^{3} V_k(I),\qquad P(I) = V(I)\,I \qquad\qquad (13)
```

Non-uniform irradiance therefore produces a multi-peak P–V curve with one global maximum power point (GMPP) and one or more local maxima (LMPPs). Within the voltage window \[V\_min, V\_max\] = \[8, 35\] V of the converter, the target of the global search is

```latex
V_{gmpp} = \arg\max_{V_{min}\le V\le V_{max}} V\,I(V),\qquad P_{gmpp} = \max_{V_{min}\le V\le V_{max}} V\,I(V) \qquad\qquad (14)
```

The lower bound of 8 V keeps within reach the peak that appears when only part of the string conducts. A single irradiance sensor on the module cannot resolve the pattern; it reports the cell-weighted mean irradiance

```latex
G_{in} = \frac{1}{N_s}\sum_{k=1}^{3}\left[(18 - n_k)\,G + n_k f_k G\right] \qquad\qquad (15)
```

### Irradiance and temperature data

Training and evaluation use real hourly irradiance G and ambient temperature T from the PVGIS-SARAH3 reanalysis dataset for Ramadi, Iraq (33.42° N, 43.30° E), spanning 2005–2020 (140,256 hourly records). Night hours (G = 0) make up 51.9% of the series and are excluded from training and scoring. The 61,738 daytime hours (G ≥ 50 W/m²) contain 61,231 distinct (G, T) pairs across the full seasonal and diurnal range (daytime mean irradiance 516 W/m² and maximum 1090 W/m²; ambient temperature from −2.9 to 50.0 °C, overall mean 24.4 °C). To prevent leakage, the data are split chronologically (Fig. 4a, Table 3): every sample pairs one real hourly condition from the years of its subset with an independently drawn plant and shading pattern, so that no hour, and no year, is shared between training and test data.

&#91;image: Fig. 4\]

**Fig. 4** Data and training workflow. (a) Chronological partition of the 2005–2020 series. (b) Real daytime conditions are combined with randomized plants and shading patterns in the shaded-string simulator, which returns the 14 probe currents and the label V\_gmpp; the DNN is trained on the training subset, its intervals are calibrated on a separate subset, and it is evaluated on held-out test years.

| Subset | Years | Use | Samples |
| --- | --- | --- | --- |
| Training | 2005–2014 | Fitting the DNN and the input and output scalers | 200,000 |
| Validation | 2015–2016 | Early stopping and learning-rate schedule | 15,000 |
| Calibration | 2017–2018 | Split-conformal quantile | 15,000 |
| Test | 2019–2020 | Offline test sets T1–T4 and the two real days of the closed-loop scenarios | 8,000 per set (16,000 for T1) |

**Table 3** Chronological partition of the PVGIS data and the samples generated from each subset.

### Hybrid scan + P&O–DNN controller

The proposed controller operates in two stages, summarized in the flowchart of Fig. 5. Stage 1 is a global search that locates the neighborhood of the global maximum; Stage 2 is a local refinement that finds the exact maximum power point (MPP) and follows it between scans.

**Stage 1: global scan.** A fast open-loop I–V scan samples the module current at 14 probe voltages evenly spaced over the converter window (about 2.08 V apart),

```latex
V_j = V_{min} + (j - 1)\,\frac{V_{max} - V_{min}}{13},\qquad j = 1,\ldots,14 \qquad\qquad (16)
```

```latex
I_j = I(V_j),\qquad P_j = V_j\,I_j \qquad\qquad (17)
```

The scan is executed within one control cycle, and the energy delivered in that cycle is taken as the mean power over the probes, so that the scan has an explicit energy cost:

```latex
P_{scan} = \frac{1}{14}\sum_{j=1}^{14} P_j \qquad\qquad (18)
```

In the non-ML variant, the probe with the highest measured power is the estimate of the global-maximum voltage,

```latex
\hat V_{bp} = V_{j^*},\qquad j^* = \arg\max_{j} P_j \qquad\qquad (19)
```

In the DNN variants, the 14 probe currents, together with the cell-weighted mean irradiance G\_in and the ambient temperature T\_amb, form the 16 inputs of the DNN, which returns the estimate V̂\_gmpp (Eq. 25). The estimate becomes the new operating voltage (clipped to the converter window), which is the handoff to Stage 2.

&#91;image: Fig. 5\]

**Fig. 5** Flowchart of the hybrid controller. Stage 1 performs the I–V scan and estimates the global-maximum voltage with either the DNN or the best probe; the estimate is the starting point of the P&O refinement of Stage 2 (omitted in the DNN scan-only variant). A drop of P/G, or the expiry of the scan period, returns the controller to Stage 1.

**Stage 2: local refinement.** Starting from the voltage handed over by Stage 1, a standard P&O loop (Eq. 37) moves the operating voltage in steps of 0.5 V per control cycle to the precise MPP and tracks it as irradiance changes.

**Re-scan rule.** A new scan is triggered when the normalized power P/G falls more than 10% below its running maximum since the last scan, at least 30 cycles after that scan, or when 300 cycles have elapsed since it, whichever occurs first:

```latex
\left[\frac{P_k}{G_k} < (1 - 0.10)\max_{k_s \le i \le k}\frac{P_i}{G_i}\ \ \text{and}\ \ k - k_s \ge 30\right]\ \ \text{or}\ \ k - k_s \ge 300 \qquad\qquad (20)
```

where k\_s is the cycle of the last scan. Only a drop triggers a scan, because a rise of P/G means that P&O is still climbing; the drop of a quantity that the irradiance sensor does not explain signals shading or a changed P–V curve. The measured energy cost of scanning is 0.15% of the total harvested energy and is included in every efficiency figure reported in Results.

Figure 6 shows the scan on a shaded string (G = 800 W/m², T\_amb = 25 °C, nominal plant, substring irradiance 800, 640 and 320 W/m²). The global maximum (95.7 W at 15.7 V) lies at a lower voltage than the local maximum (81.6 W at 26.4 V); a third maximum at 6.6 V lies below the converter window. A P&O loop started at 26 V would settle on the local maximum, at 85% of the global power. The scan identifies the correct region (best probe at 16.3 V, 94.2 W), and P&O then removes the remaining 0.6 V of error.

&#91;image: Fig. 6\]

**Fig. 6** I–V scan on a partially shaded string (simulator output, nominal plant). The 14 probes sample the P–V curve within the converter window (grey); the best probe lies close to the global maximum, which is the regression target of the DNN, and far from the local maximum at which a hill-climbing method started at 26 V would stop.

Three configurations of the global-search stage are evaluated to isolate the contribution of the DNN:

- **Scan+P&O (non-ML):** the best probe of Eq. (19) is used directly, with no learned model.
- **DNN scan-only:** the DNN estimate is held until the next scan, without P&O refinement.
- **Hybrid (proposed):** the DNN estimate initializes the P&O refinement stage.

### Training data and domain randomization

A network trained on one nominal plant learns that plant, not the task. To avoid overfitting to a single plant, and to expose the network to realistic partial-shading patterns during training rather than only at test time, every training sample has its own plant and its own shading pattern, paired with a real PVGIS hourly condition (Fig. 4b). The electrical parameters of each plant are drawn independently around the datasheet extraction of Table 2,

```latex
p = p_{nom}\,u_p,\quad u_p \sim \mathcal{U}(l_p, h_p),\quad p \in \{I_{sc}, V_{oc}, R_s, R_{sh}\};\qquad n \sim \mathcal{U}(1.05, 1.40) \qquad\qquad (21)
```

with the ranges of Table 4, which represent manufacturing tolerance and aging. The saturation and light-generated currents of each plant follow in closed form from Eqs. (8) and (9), so that every plant satisfies its own I\_sc and V\_oc exactly.

| Parameter | Nominal | Training range | Out-of-range test set T4 |
| --- | --- | --- | --- |
| I\_sc | 9.87 A | ×0.90 – ×1.05 | ×0.78 – ×0.86 |
| V\_oc | 32.9 V | ×0.95 – ×1.06 | ×1.08 – ×1.12 |
| R\_s | 0.182 Ω | ×0.8 – ×2.4 | ×2.8 – ×3.8 |
| R\_sh | 400 Ω | ×0.3 – ×1.5 | ×0.12 – ×0.25 |
| n | 1.2 | 1.05 – 1.40 (absolute) | 1.45 – 1.55 (absolute) |

**Table 4** Domain-randomization ranges of the plant parameters (multipliers of the nominal value, except n) and the ranges of the out-of-range test set T4, which lies entirely outside the training box.

Sixty percent of the samples are shaded. For a shaded sample, each of the three substrings is shaded independently with probability 0.5 (at least one substring is always shaded); a shaded substring has either all 18 cells shaded (probability 0.5) or a uniformly drawn number of 1 to 17 cells shaded, and the shaded cells receive the fraction f\_k ∼ U(0.10, 0.85) of the irradiance. The shaded string follows the bypass-diode model of Eqs. (11)–(13), and agrees with an independent shaded-string implementation to three decimals.

Generation of the 200,000 training samples used parallel simulation across 4 worker processes, with results bit-identical to serial execution. On a 12,000-sample benchmark the measured speedup was 2.1× in the reported run (1.3× in another execution on a loaded machine); generating all 200,000 samples took about 20 s of wall-clock time (about 72 s serial-equivalent). The workload is embarrassingly parallel: each sample is independent, and the cost grows linearly with the number of plants and of randomized parameters. This is the sense in which the training of the estimator is relevant to parallel and high-performance computing, while the resulting controller is lightweight at inference time (see Results).

### DNN architecture and forward computation

**Task.** The estimator is a regression function f\_θ that maps the 16-dimensional input x = (G\_in, T\_amb, I\_1, …, I\_14) to the global-maximum voltage y = V\_gmpp of Eq. (14). The probe currents carry the information about the shape of the P–V curve: every substring that is shaded or unshaded introduces a step in the I–V curve at its own photocurrent level, and the positions and sizes of these steps determine which peak is the global one. The mean irradiance and the ambient temperature provide the scale of the problem. The inputs and the target are standardized with the mean and standard deviation of the training subset only,

```latex
\tilde x_i = \frac{x_i - \mu_i}{\sigma_i},\qquad \tilde y = \frac{y - \mu_y}{\sigma_y} \qquad\qquad (22)
```

**Forward pass.** The network is a fully connected feed-forward network with three hidden layers of 128, 128 and 64 neurons and one linear output neuron (Fig. 7). Each hidden layer l computes

```latex
h^{(l)} = \phi\!\left(W^{(l)}\,h^{(l-1)} + b^{(l)}\right),\qquad h^{(0)} = \tilde x,\qquad l = 1,2,3 \qquad\qquad (23)
```

where W^(l) is the weight matrix and b^(l) the bias vector of the layer, and φ is the rectified linear unit (ReLU),

```latex
\phi(z) = \max(0, z) \qquad\qquad (24)
```

Each neuron therefore forms a weighted sum z of the outputs of the previous layer plus a bias and passes it through φ. The output layer is linear, and the prediction is converted back to volts and limited to the converter window:

```latex
\hat y = W^{(4)}h^{(3)} + b^{(4)},\qquad \hat V_{gmpp} = \min\left\{\max\left[\sigma_y\,\hat y + \mu_y,\ V_{min}\right],\ V_{max}\right\} \qquad\qquad (25)
```

&#91;image: Fig. 7\]

**Fig. 7** Architecture of the hybrid stage-1 DNN: 16 standardized inputs (cell-weighted mean irradiance, ambient temperature and the 14 probe currents), three ReLU hidden layers of 128, 128 and 64 neurons, and one linear output (the standardized global-maximum voltage). Only a subset of the neurons of each layer is drawn; every neuron is connected to all neurons of the next layer.

The weight matrices have the dimensions 128×16, 128×128, 64×128 and 1×64. The number of trainable parameters and the number of multiply–accumulate operations per inference follow from the layer widths n\_0 = 16, n\_1 = 128, n\_2 = 128, n\_3 = 64 and n\_4 = 1:

```latex
N_\theta = \sum_{l=1}^{4}(n_{l-1} + 1)\,n_l = 2176 + 16512 + 8256 + 65 = 27009 \qquad\qquad (26)
```

```latex
N_{MAC} = \sum_{l=1}^{4} n_{l-1}\,n_l = 2048 + 16384 + 8192 + 64 = 26688 \qquad\qquad (27)
```

The layers are 16 → 128 → 128 → 64 → 1 with ReLU activations on the hidden layers. In single precision the parameters occupy about 108 kB.



**Design rationale.** The design rests on four choices.

- **Feed-forward regression network.** The target is a single continuous quantity that is a deterministic function of the 16 inputs, and labeled examples are available in any number from the simulator. A supervised regression network uses these labels directly, has a fixed and small inference cost and, unlike a fuzzy-logic controller or a reinforcement-learning agent, needs no hand-designed rule base and no online exploration. A fuzzy system with 16 inputs would need an unmanageable number of rules or would have to discard most inputs, and its membership functions would have to be tuned by hand or by a separate optimizer; a reinforcement-learning agent needs a reward design, many closed-loop episodes and a safe way to explore on real hardware, none of which is needed for a quantity that can be labeled offline.
- **ReLU activation.** The function φ is cheap to evaluate, its gradient is 0 or 1 (so that gradients do not vanish through three hidden layers), and a network built from it is piecewise linear, which suits the piecewise character of the I–V curves of a shaded string.
- **Three hidden layers of decreasing width.** Two layers of 128 neurons expand the 16 inputs into a richer representation, and a layer of 64 neurons condenses it before the output. With 27,009 parameters and 200,000 training samples, there are about 7.4 samples per parameter, and every sample is a distinct combination of plant, shading pattern and hour, so that the network cannot memorize a single curve. The depth and widths were fixed a priori and were not tuned on the test years; a formal architecture search was not performed (see Discussion).
- **No dropout, batch normalization or weight decay.** At this ratio of samples to parameters, and with early stopping, no explicit regularizer was needed: the validation loss follows the training loss closely (Results), and the model is simple enough for its behavior to be audited.

The inference cost is 26,688 multiply–accumulate operations, which is small enough for the memory of a low-cost microcontroller-class device (about 108 kB in single precision, less with 8-bit quantization); no embedded implementation was made in this study, and the measured cost per control cycle is reported in Results.

### Training procedure

The network is trained to minimize the mean squared error of the standardized target over mini-batches of B = 1,024 samples,

```latex
\mathcal{L}(\theta) = \frac{1}{B}\sum_{i=1}^{B}\left(\hat y_i - \tilde y_i\right)^2 \qquad\qquad (28)
```

The gradients are obtained by backpropagation and the parameters are updated with the Adam optimizer [27], with the initial learning rate η = 10⁻³ and the default constants β₁ = 0.9, β₂ = 0.999 and ε = 10⁻⁷. The weights were initialized with the Glorot-uniform scheme and the biases with zero, with a fixed random seed of 42. After every epoch the loss is evaluated on the validation subset (Table 3), and two rules act on it:

```latex
\eta \leftarrow 0.5\,\eta\ \ \text{after 4 epochs without improvement } (\eta \ge 10^{-6});\qquad \text{stop after 8 epochs without improvement, restoring the best weights} \qquad\qquad (29)
```

The first rule refines the step size when progress stalls; the second stops training before the network starts to fit noise in the training set. The maximum number of epochs was 60, and training stopped after 48 epochs. Fitting took 44 s, so that the whole pipeline (data generation and training) needed about 64 s on four CPU cores.

### Offline evaluation

Offline accuracy was measured on four held-out sets from 2019–2020: T1, randomized plants inside the training range with mixed shading; T2, the nominal plant under uniform irradiance; T3, the nominal plant under partial shading; and T4, plants whose parameters lie outside the randomization box (Table 4), with mixed shading. For n test samples with true voltage y\_i and prediction ŷ\_i (both in volts, not standardized), the accuracy of the voltage estimate is measured by

```latex
\begin{aligned}\mathrm{MSE} &= \frac{1}{n}\sum_{i=1}^{n}\left(y_i - \hat y_i\right)^2,\qquad \mathrm{RMSE} = \sqrt{\mathrm{MSE}},\\ \mathrm{MAE} &= \frac{1}{n}\sum_{i=1}^{n}\left|y_i - \hat y_i\right|\end{aligned} \qquad\qquad (30)
```

```latex
R^2 = 1 - \frac{\sum_{i=1}^{n}\left(y_i - \hat y_i\right)^2}{\sum_{i=1}^{n}\left(y_i - \bar y\right)^2} \qquad\qquad (31)
```

where ȳ is the mean of the true values; R² is reported for the plain ANN-MPPT models. Because a small voltage error near a flat maximum costs little power, while a wrong-peak error costs a lot, the stage-1 estimate is also scored by the power it would deliver before any P&O refinement,

```latex
\rho = \frac{100\%}{n}\sum_{i=1}^{n}\frac{P_i(\hat V_i)}{P_{gmpp,i}} \qquad\qquad (32)
```

where P\_i(V) is the power of the i-th simulated string at voltage V (linear interpolation of its curve). The DNN is compared with the non-ML best-probe rule of Eq. (19) on the same samples.

### Uncertainty quantification

Rather than an uncalibrated ensemble-spread heuristic, prediction intervals are produced with split conformal prediction [28]. On the calibration subset (n = 15,000 samples that were used neither for training nor for early stopping), the absolute residuals in volts are

```latex
r_i = \left|y_i - \hat y_i\right|,\qquad i = 1,\ldots,n \qquad\qquad (33)
```

and, for a miscoverage level α, the conformal quantile is the ⌈(n + 1)(1 − α)⌉-th smallest residual,

```latex
\hat q_\alpha = r_{\left(\lceil (n+1)(1-\alpha)\rceil\right)} \qquad\qquad (34)
```

The prediction interval for a new input x is

```latex
C_\alpha(x) = \left[\hat V_{gmpp}(x) - \hat q_\alpha,\ \hat V_{gmpp}(x) + \hat q_\alpha\right] \qquad\qquad (35)
```

If the calibration and test samples are exchangeable, this interval contains the true voltage with probability of at least 1 − α, whatever the network. Levels of 0.90 and 0.95 are used. The empirical coverage on n\_test held-out samples,

```latex
\mathrm{Cov} = \frac{1}{n_{test}}\sum_{i=1}^{n_{test}}\mathbb{1}\left[y_i \in C_\alpha(x_i)\right] \qquad\qquad (36)
```

is verified against the nominal level (Results, Table 9). The guarantee depends on exchangeability: for plants that lie outside the training distribution it is not expected to hold, and the coverage measured on such plants is reported in Results.

### Baselines and evaluation protocol

The hybrid controller is compared with standalone P&O, standalone incremental conductance (IncCond), a scan-based non-ML baseline with the same sensors as the hybrid (Scan+P&O), a particle-swarm global tracker (PSO), a scan-based variant in which a gradient-boosted-tree regressor replaces the DNN (Tree scan+P&O), a quadratic polynomial fitted to (G, T) \[“Poly2(G,T)”\], and a plain ANN-MPPT controller without a scan stage. Table 5 lists the controllers and the signals each needs. P&O, IncCond and PSO use only the PV voltage and current; Poly2(G,T) and the plain ANN-MPPT also use irradiance and temperature sensors, and the scan-based controllers additionally require a converter able to perform a fast I–V scan. Because of this asymmetry, the equal-information comparison for the ANN is Poly2(G,T), not P&O or IncCond. The selection follows the role of each method. P&O and IncCond are the most widely used conventional trackers and therefore the reference that any new controller must beat; the plain ANN-MPPT is the most common learning-based tracker in the literature and shows what a network achieves without a global search; Poly2(G,T) is its equal-information, non-neural counterpart; Scan+P&O has the same structure as the hybrid without a network and so isolates the contribution of the DNN; and PSO represents the metaheuristic global-search family.

| Controller | Signals | Decision rule | Settings |
| --- | --- | --- | --- |
| P&O | V, I | Eq. (37) | Step 0.5 V; dead band 0.01 W; start 26 V |
| IncCond | V, I | Eqs. (38) and (39) | Step 0.5 V; tolerances 0.001 S and 0.1 V |
| Poly2(G,T) | G, T | V\_ref from a ridge-regularized quadratic polynomial of (G, T) fitted to the MPP voltage of the nominal plant; Eq. (40) | Slew limit 0.5 V per cycle |
| ANN-MPPT | G, T, V, I | V\_ref from a fully connected network (64, 64, 32 neurons); Eq. (40) | Trained on the nominal plant only; slew limit 0.5 V per cycle |
| Scan+P&O | I–V scan | Best probe, Eq. (19), then P&O, Eq. (37) | Re-scan rule, Eq. (20) |
| DNN scan-only | I–V scan, G\_in, T\_amb | DNN estimate, Eq. (25), held until the next scan | Re-scan rule, Eq. (20) |
| Hybrid (proposed) | I–V scan, G\_in, T\_amb | DNN estimate, Eq. (25), then P&O, Eq. (37) | Re-scan rule, Eq. (20) |
| PSO | V, I | Four particles (voltage setpoints) evaluated one per cycle, velocity update after each full sweep of the swarm; Fig. 9 | w = 0.4; c1 = c2 = 1.5; converged when all particles are within 0.3 V; restart at \|ΔP\|/P > 10% or after 600 cycles |
| Tree scan+P&O | I–V scan, G\_in, T\_amb | Gradient-boosted-tree estimate of V\_gmpp from the same 16 inputs, then P&O, Eq. (37) | 31 leaves; learning rate 0.1; iteration count chosen on 2015–2016; re-scan rule, Eq. (20) |

**Table 5** Compared controllers, the signals they use and their settings. All controllers share the 8–35 V voltage window and the 0.5 V per cycle actuator limit, except where explicitly noted (the PSO tracker moves directly between particle positions without the slew limit).

The P&O rule (Fig. 8a) compares the power and voltage of two consecutive cycles, ΔP\_k = P\_k − P\_{k−1} and ΔV\_k = V\_k − V\_{k−1}, and keeps or reverses the perturbation direction d\_k:

```latex
d_k = \begin{cases} d_{k-1}, & |\Delta P_k| \le 0.01\ \mathrm{W}\\ \operatorname{sign}(\Delta V_k), & \Delta P_k > 0.01\ \mathrm{W}\\ -\operatorname{sign}(\Delta V_k), & \Delta P_k < -0.01\ \mathrm{W}\end{cases}\qquad V_{k+1} = \min\left\{\max\left[V_k + d_k\,\Delta V_{step},\ V_{min}\right],\ V_{max}\right\} \qquad\qquad (37)
```

with ΔV\_step = 0.5 V. The IncCond rule (Fig. 8b) exploits the fact that, at the MPP, dP/dV = I + V·dI/dV = 0, so that the incremental conductance dI/dV equals the negative of the instantaneous conductance −I/V. The deviation from this condition is

```latex
C_k = \frac{\Delta I_k}{\Delta V_k} + \frac{I_k}{V_k} \qquad\qquad (38)
```

which is positive to the left of the MPP and negative to its right. For |ΔV\_k| > 0.1 V the voltage step is chosen as

```latex
\Delta V_{step,k} = \begin{cases} +0.5\ \mathrm{V}, & C_k > \varepsilon_C\\ 0, & |C_k| \le \varepsilon_C\\ -0.5\ \mathrm{V}, & C_k < -\varepsilon_C\end{cases}\qquad \varepsilon_C = 0.001\ \mathrm{S} \qquad\qquad (39)
```

and, when the voltage has hardly changed (|ΔV\_k| ≤ 0.1 V), the step follows the sign of ΔI\_k if |ΔI\_k| > 0.001 A and is zero otherwise. The sensor-based controllers (Poly2 and ANN-MPPT) obtain a reference voltage V\_ref from their predictor and move toward it under the same actuator limit,

```latex
V_{k+1} = V_k + \min\left\{\max\left[V_{ref,k} - V_k,\ -0.5\right],\ +0.5\right\}\ \mathrm{V} \qquad\qquad (40)
```

&#91;image: Fig. 9\]

**Fig. 8** Decision logic of the two conventional baselines as implemented. (a) Perturb and observe (Eq. 37). (b) Incremental conductance (Eqs. 38 and 39).

The PSO tracker is a model-free global-search comparator that needs neither a scan nor irradiance and temperature sensors. Four particles, each a voltage setpoint in the 8–35 V window, are evaluated one per control cycle. After the whole swarm has been evaluated, the velocity of each particle is updated as v ← w v + c1 r1 (p\_best − x) + c2 r2 (g\_best − x), with r1 and r2 uniform on (0, 1), and the position as x ← x + v, with the velocity limited to 25% of the voltage window. The swarm is considered converged when all particles lie within 0.3 V of the best position, which is then held; the swarm is re-initialized when the power deviates from the held value by more than 10% or after 600 cycles. The inertia weight w = 0.4 and the acceleration coefficients c1 = c2 = 1.5 were fixed a priori and not tuned on any scenario.



In the tree-based variant, a histogram gradient-boosted regression-tree ensemble with 31 leaves per tree and a learning rate of 0.1 replaces the DNN as the stage-1 estimator. It receives the same 16 inputs, is trained on the same 200,000 plant samples, and its number of boosting iterations was chosen on the 2015–2016 validation years; the scan, the hand-off to P&O and the re-scan rule are identical to those of the hybrid. This variant tests whether the benefit of the hybrid depends on the neural network. Table 6 collects the fixed parameters of all algorithms.

**Table 6** Parameters of the compared algorithms. All were fixed a priori; none was tuned on the 2019–2020 test years.

| Controller | Parameter | Value |
| --- | --- | --- |
| All | Control cycle; converter voltage window; start voltage | 1 s; 8–35 V; 26 V |
| P&O, IncCond, scan hand-off | Perturbation step | 0.5 V per cycle |
| P&O | Minimum power change | 0.01 W |
| IncCond | Voltage tolerance; conductance tolerance | 0.1 V; 0.001 S |
| Poly2(G,T) | Model | Quadratic polynomial of (G, T), ridge 1e-6, 6 coefficients |
| ANN-MPPT | Network; inputs; output | (64, 64, 32) hidden units; (G, T, V, I); V\_ref, slew 0.5 V per cycle |
| Scan-based controllers | Probe voltages (14) | 8.00, 10.08, 12.15, 14.23, 16.31, 18.38, 20.46, 22.54, 24.62, 26.69, 28.77, 30.85, 32.92, 35.00 V |
| Scan-based controllers | Re-scan trigger | P/G below 90% of its running maximum since the last scan (at least 30 cycles after it), or every 300 cycles |
| Hybrid, DNN scan-only | Stage-1 DNN | Inputs (G\_in, T, I\_1…I\_14); hidden (128, 128, 64), ReLU; output V\_gmpp; Adam 1e-3, batch 1024, at most 60 epochs, early stopping with patience 8 on 2015–2016 |
| PSO | Swarm | 4 particles; w = 0.4; c1 = c2 = 1.5; convergence within 0.3 V; restart when the power deviates from the held value by more than 10% or after 600 cycles |
| Tree scan+P&O | Estimator | Histogram gradient-boosted trees; 31 leaves; learning rate 0.1 |
| Metrics | Convergence; GMPP success; wrong peak | 2% for 10 cycles; 2% over the last 10 non-scan cycles; error > 2 V |

All controllers are evaluated on an identical set of scenarios. The standard scenarios are an irradiance step from 400 to 800 W/m², periodic cloud passing (400–1200 W/m²), a linear ramp of irradiance and temperature, and a comprehensive scenario that combines cloud-like variation with a temperature drift. Two real days from the test years, a clear summer day (24 June 2020) and a variable day (6 April 2020), are time-compressed to 3,600 control cycles. Nine partial-shading patterns of 400 s each are applied to the three-substring string; they were fixed before any controller was run and are not in the training list, although they belong to the same pattern family. A pattern counts as solved when the mean power over the last 10 non-scan cycles of its episode reaches 98% of the global maximum. Cold-start tests begin from five valid initial voltages between 13.4 and 28.9 V at 800 W/m² and 25 °C; an extended protocol with initial voltages of 15, 20, 26, 30 and 35 V (the last two close to or above the open-circuit voltage) is reported separately. Plant mismatch is introduced by perturbing the evaluated plant relative to the plant used for training. For a severity s, the light-generated current, the series resistance, the shunt resistance and the ideality factor are changed to

```latex
I_{ph}' = (1 - 0.08\,s)\,I_{ph},\quad R_s' = (1 + 0.8\,s)\,R_s,\quad R_{sh}' = (1 - 0.4\,s)\,R_{sh},\quad n' = (1 + 0.05\,s)\,n \qquad\qquad (41)
```

so that severity 1 reduces I\_ph by 8%, increases R\_s by 80%, reduces R\_sh by 40% and increases n by 5%. The sweep covers s = 0 to 2; severities 1.5 and 2 lie outside the randomization range of the DNN (Table 4).

**Boost-converter plant.** Besides the quasi-static plant, in which the PV operating point equals the commanded voltage, the controllers were run on a plant that includes a DC–DC boost converter (**Fig. 9**). The PV source feeds an input capacitor C\_in and an inductor L (series resistance R\_L), a low-side switch (on-resistance R\_on) and a diode (forward drop V\_f) towards a constant bus V\_bus. In the averaged continuous-conduction model with duty cycle d, the PV voltage v\_pv and the inductor current i\_L obey C\_in dv\_pv/dt = i\_pv(v\_pv) − i\_L and L di\_L/dt = v\_pv − i\_L (R\_L + d R\_on) − (1 − d)(V\_bus + V\_f). An inner PI loop on the PV voltage with feed-forward, d = 1 − v\_pv/(V\_bus + V\_f) + K\_p e + K\_i ∫e dt with e = v\_pv − V\_cmd and d limited to 0.05–0.95, follows the command V\_cmd of the MPPT algorithm. The MPPT algorithm sees the mean voltage, current and power over the last quarter of a 20 ms settling window, which is extended to 60 ms when the command differs from the present voltage by more than 2 V (scan probes and start-up). A switching-level model (PWM at f\_sw with sampled duty, 100 steps per period, diode with a discontinuous-conduction clamp) was used to verify the averaged model. The parameters (Table 7) correspond to a 250 W module-level converter: L = 1 mH gives an inductor ripple of about 0.6 A (7% of 9 A) at the nominal operating point (26 V, d = 0.46, 20 kHz), C\_in = 470 µF an input-voltage ripple of about 8 mV, and the PI gains were fixed once on the step response at 800 W/m² and 25 °C before any scenario was simulated. A 14-point scan needs at most 14 × 60 ms = 0.84 s and fits inside one 1 s control cycle.

&#91;image: Fig. 11\]

**Fig. 9** System overview: PV module or shaded string, input capacitor, boost converter to a constant bus, voltage and current sensing, MPPT controller (command V\_cmd) and inner voltage loop (duty D).

**Table 7** Boost-converter parameters (averaged and switching-level models).

| Parameter | Value |
| --- | --- |
| Inductance L; series resistance R\_L | 1 mH; 20 mΩ |
| Input capacitance C\_in | 470 µF |
| Switching frequency f\_sw | 20 kHz |
| Bus voltage V\_bus | 48 V |
| Diode forward drop V\_f; switch on-resistance R\_on | 0.5 V; 15 mΩ |
| PI gains K\_p; K\_i | 0.05 1/V; 10 1/(V s) |
| Duty-cycle limits | 0.05–0.95 |
| Settling window per measurement | 20 ms (60 ms for command jumps larger than 2 V) |
| Averaged-model step; switching-model steps per PWM period | 20 µs; 100 |


### Metrics

For every controller and scenario, five metrics are reported. The **tracking efficiency** is the harvested energy as a percentage of the theoretical maximum available energy over a run of N control cycles of duration Δt, including any scan overhead:

```latex
\eta_{\mathrm{track}} = \frac{\sum_{k=1}^{N} P_{\mathrm{out}}(k)\,\Delta t}{\sum_{k=1}^{N} P_{\mathrm{mpp}}(k)\,\Delta t} \times 100\% \qquad\qquad (42)
```

where P\_out(k) is the power harvested in cycle k (in a scan cycle, the mean probe power P\_scan of Eq. (18)) and P\_mpp(k) is the true global maximum power available at the prevailing irradiance, temperature and shading pattern. The **convergence time** is the number of control cycles after a scenario event (start, irradiance step or shading transition at cycle k\_e) until the power first stays within 2% of the true maximum for 10 consecutive cycles:

```latex
k_{conv} = \min\left\{k \ge k_e:\ \frac{P_j}{P_{mpp,j}} \ge 0.98\ \ \text{for all}\ j = k,\ldots,k+9\right\} - k_e \qquad\qquad (43)
```

The **steady-state voltage oscillation amplitude** is the peak-to-peak voltage over a window W of 10 consecutive cycles taken at least 50 cycles after the latest event:

```latex
A_{osc} = \max_{k\in W} V_k - \min_{k\in W} V_k \qquad\qquad (44)
```

The **partial-shading success rate** is the fraction of the nine shading patterns that a controller solves, a pattern being solved when the mean power over the last 10 non-scan cycles Q of its episode reaches 98% of the global maximum, that is, when it reaches the GMPP rather than an LMPP:

```latex
S = \frac{1}{9}\sum_{m=1}^{9}\mathbb{1}\left[\frac{1}{|Q|}\sum_{k\in Q}P_k \ge 0.95\,P_{gmpp,m}\right] \qquad\qquad (45)
```

Finally, the **computational cost per control cycle** is the measured wall-clock execution time of the decision code (Python and numpy on a CPU, plant model excluded). All controllers obey the same 0.5 V per control cycle actuator limit, except where explicitly noted: the cold-start protocol separately reports a scan-based result that assumes that the converter can execute the initial I–V sweep and the voltage handoff without this limit (see Results).

## Results

All results come from simulation. The plant is the single-diode module or shaded-string model of the Methodology, driven by PVGIS-SARAH3 data and by synthetic scenarios, and it is quasi-static (the PV operating point responds instantly to the commanded voltage) unless a boost-converter plant is named explicitly. Controllers are scored on the held-out years 2019–2020 and on synthetic scenarios only; no parameter was tuned on these data. Efficiencies are energy-weighted tracking efficiencies and include the scan overhead. The nominal plant is the model used to generate the training data, the mismatched plant is the perturbed model of severity 1 defined in the Methodology, and the converter plant is the averaged boost converter of Table 7. The headline numbers are single deterministic runs; their seed-to-seed spread and paired significance tests are given in the subsection on statistical robustness. Every figure has a data file and every table a CSV in the public repository (Declarations), and the numbers quoted in the text are read from those files. Timing values depend on the machine (Table 15) and are reported with that caveat.

### Data and PV model

The PVGIS-SARAH3 series for Ramadi is characterized by the following statistics. Night hours make up 51.9% of the 140,256 records, and the 61,738 daytime hours have a mean irradiance of 516 W/m² and a maximum of 1090 W/m². Ambient temperature averages 24.4 °C, ranges from −2.9 to 50.0 °C and is bimodal because of the hot summer and cool winter. Irradiance and temperature are only moderately correlated (r = 0.42), and irradiance above 900 W/m² occurs almost only in the warmer half of the year, so cell temperatures of 60–79 °C are reached at the highest irradiances.







The single-diode model reproduces the datasheet-type inputs at standard test conditions (I\_sc = 9.87 A, V\_oc = 32.9 V and P\_max = 244.60 W at 26.60 V; **Fig. 10**), and its temperature coefficients emerge at −0.33 %/K for V\_oc and −0.41 %/K for P\_max without having been imposed. Raising the cell temperature from 25 to 65 °C at 1000 W/m² lowers V\_oc from 32.9 to 28.5 V and P\_max from 245 to 204 W. At fixed cell temperature, V\_mpp falls by 2.1%, 5.0% and 8.6% relative to STC at 400, 200 and 100 W/m², and by 0.42 %/K of its STC value between 25 and 45 °C, so temperature remains the dominant factor but irradiance cannot be neglected.



&#91;image: Fig. 16\]

**Fig. 10** Validation of the single-diode model at STC against the datasheet-type inputs: (a) I–V curve and (b) P–V curve.

### Stage-1 DNN: accuracy and uncertainty

The DNN trained for 48 epochs before early stopping on the 2015–2016 validation years, and its validation loss follows the training loss closely, so there is no sign of overfitting. On the in-distribution test set its RMSE for the global-maximum voltage is 0.78 V and its MAE is 0.28 V (headline seed 42; over five seeds the RMSE is 0.75 ± 0.03 V, Table S3). The residuals are centered on zero but heavy-tailed (**Fig. 11b**): most errors are a few tenths of a volt, while rare wrong-peak choices on curves with two peaks of similar height produce errors of several volts. In the closed loop these outliers are corrected by the P&O stage and by the periodic re-scan.



&#91;image: Fig. 18\]

**Fig. 11** Stage-1 DNN on the in-distribution test set: (a) estimated against true global-maximum voltage and (b) histogram of the estimation error; RMSE 0.78 V.

Table 8 compares the stage-1 estimate with three alternatives that use the same inputs: a gradient-boosted-tree regressor, a linear (ridge) regressor and the non-ML best-probe rule (take the probe voltage with the highest measured power), by the power ratio P(V\_est)/P\_gmpp before any P&O refinement. The DNN has the lowest RMSE in distribution and the highest power ratio on uniform irradiance (99.97% versus 99.50% for the best-probe rule) and on shaded curves of the nominal plant (98.77% versus 98.64%). It is not better in every respect. Its wrong-peak rate in distribution (2.1%) is higher than that of the best-probe rule (0.6%), because a regression output can fall between two peaks while the best-probe rule always returns a measured point. For plants outside the randomization box the DNN is worse than the simple rule (98.44% versus 99.37%). The linear regressor is clearly inadequate (RMSE 3.2 V), and the tree regressor is less accurate than the DNN offline (RMSE 1.21 V) although it performs as well in the closed loop (see the comparators below).

**Table 8** Offline accuracy of the stage-1 estimate on held-out 2019–2020 data: parameters, multiply–accumulate operations (MACs), RMSE of the global-maximum voltage, and mean power ratio P(V\_est)/P\_gmpp before P&O refinement, for the in-distribution set (T1), the nominal plant under partial shading (T3) and out-of-range plants (T4). The first row is the headline DNN (seed 42).

| Estimator | Params | MACs | RMSE T1 (V) | RMSE T3 (V) | RMSE T4 (V) | Power ratio T1 (%) | Power ratio T3 (%) | Power ratio T4 (%) | Wrong peak T1 (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DNN (128,128,64) | 27,009 | 26,688 | 0.78 | 0.94 | 1.67 | 99.22 | 98.77 | 98.44 | 2.11 |
| Gradient-boosted trees | 24,400 | 2,000 | 1.21 | 1.47 | 1.81 | 97.76 | 96.42 | 96.13 | 7.12 |
| Ridge (linear) | 17 | 16 | 3.20 | 3.87 | 3.73 | 85.89 | 78.28 | 85.32 | 51.78 |
| Best-probe rule (no ML) | 0 | 14 | 1.05 | 1.27 | 1.06 | 99.01 | 98.64 | 99.37 | 0.63 |

The split-conformal intervals are calibrated inside the training distribution (Table 9): coverage is 0.899 at the 0.90 level and 0.949 at the 0.95 level, with half-widths of 0.41 and 0.73 V. Coverage is lower on shaded samples (0.83 and 0.91) and falls to 0.41 and 0.76 for out-of-range plants, so the intervals are valid only under exchangeability and do not protect against distribution shift.

**Table 9** Empirical coverage of the split-conformal intervals of the hybrid DNN (half-widths 0.41 V at the 0.90 level and 0.73 V at the 0.95 level). T1 is the in-distribution test set, T2 the nominal plant with uniform irradiance, T3 the nominal plant under partial shading and T4 out-of-range plants.

| Test set | n | Coverage at 0.90 | Coverage at 0.95 |
| --- | --- | --- | --- |
| T1 in distribution (all) | 16,000 | 0.899 | 0.949 |
| T1, shaded samples | 9,456 | 0.833 | 0.915 |
| T1, unshaded samples | 6,544 | 0.995 | 0.999 |
| T2 nominal plant, uniform irradiance | 8,000 | 0.999 | 1.000 |
| T3 nominal plant, partial shading | 8,000 | 0.845 | 0.929 |
| T4 out-of-range plants | 8,000 | 0.407 | 0.758 |



### Partial shading

Table 10 lists the nine shading patterns and the efficiency of every controller in each. A pattern counts as solved when the mean power over the last 10 non-scan cycles of its 400-cycle episode is at least 98% of the global maximum. P&O, IncCond, Poly2(G,T) and the plain ANN solve three of the nine patterns (33.3%); in the other six they remain on a local peak at higher voltage and lose between 10% and 57% of the available power. The three patterns that every controller solves have uniform irradiance or a global peak at high voltage. Scan+P&O, the DNN scan-only variant, the hybrid and the tree-based variant reach the global maximum in all nine patterns, and so does PSO (**Fig. 12** shows the end points in the P–V plane, **Fig. 13** the time response). The DNN scan-only variant, which applies its estimate without a P&O refinement, is the weakest of the scan-based controllers in pattern 6 (97.4%), where the estimate misses the peak by a few tenths of a volt; the hybrid recovers this loss (99.4%), which is the role of the P&O stage.

**Table 10** Tracking efficiency (%) in each of the nine partial-shading patterns (irradiance of the three substrings in W/m², T = 25 °C; nominal plant, 400 control cycles per pattern). The efficiency covers the whole episode, including the transient after the pattern change, whereas the success criterion uses only the last 10 non-scan cycles; PSO ends pattern 8 on the global peak although its energy-weighted efficiency in that pattern is 79.9%.

| Pattern | Substrings (W/m²) | P&O | IncCond | Poly2 | ANN | Scan+P&O | DNN scan-only | Hybrid | Tree scan+P&O | PSO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 800, 800, 800 | 99.7 | 99.8 | 100.0 | 100.0 | 99.6 | 99.7 | 99.5 | 99.6 | 99.4 |
| 2 | 800, 800, 400 | 90.2 | 90.2 | 85.9 | 85.4 | 99.4 | 99.7 | 99.4 | 99.4 | 98.1 |
| 3 | 800, 800, 200 | 46.0 | 46.0 | 43.1 | 42.9 | 99.5 | 99.8 | 99.5 | 99.5 | 97.8 |
| 4 | 800, 500, 500 | 99.6 | 99.7 | 99.5 | 99.3 | 99.4 | 99.6 | 99.4 | 99.5 | 98.4 |
| 5 | 800, 600, 300 | 84.2 | 84.2 | 79.8 | 79.6 | 99.2 | 99.7 | 99.1 | 99.2 | 98.2 |
| 6 | 1000, 1000, 300 | 56.3 | 56.3 | 52.3 | 52.1 | 99.5 | 97.4 | 99.4 | 99.5 | 98.3 |
| 7 | 900, 700, 300 | 73.4 | 73.4 | 69.1 | 68.7 | 99.2 | 99.0 | 99.2 | 99.2 | 98.1 |
| 8 | 700, 700, 700 | 99.7 | 99.6 | 100.0 | 100.0 | 99.3 | 99.4 | 99.1 | 99.3 | 79.9 |
| 9 | 1000, 600, 200 | 56.1 | 56.1 | 52.5 | 52.3 | 99.0 | 99.6 | 99.1 | 98.9 | 97.2 |

&#91;image: Fig. 20\]

**Fig. 12** P–V curves of the nine shading patterns (T = 25 °C) with the operating point of every controller at the end of its episode; crosses mark the true global maximum. The conventional and model-based controllers stay on the high-voltage local peak in six of the nine patterns.

Over the nine patterns the energy-weighted efficiency on the nominal plant is 80.41% for P&O, 80.44% for IncCond, 78.17% for Poly2(G,T) and 77.99% for the plain ANN, against 99.36% for Scan+P&O, 99.30% for the DNN scan-only variant, 99.32% for the hybrid and 99.37% for the tree-based variant (Table 11). On the mismatched plant the values are 81.10%, 81.15%, 77.20% and 76.98% against 99.36%, 99.33%, 99.37% and 99.39% (**Table 11**). Scanning measures the actual plant, so the scan-based controllers are insensitive to the mismatch. PSO also finds the global peak in every pattern, but it needs on average 29.6 control cycles per event, against 1.75 for the hybrid, because its swarm must be re-initialized and converge from scratch; its shading efficiency is therefore 95.68%, and its voltage oscillation (2.11 V peak to peak) is the largest of all controllers. The hybrid and the non-ML scan controller are indistinguishable in these tests (Table 11, and Table S2 for the paired test): the global search, not the network, is what solves partial shading.

**Table 11** Summary of tracking efficiency (%) and global-maximum success rate (percentage of the nine shading patterns solved) of all controllers on the quasi-static plant. Standard scenarios are the average of the four synthetic scenarios (comprehensive, step, cloud, ramp), real days the average of the two held-out PVGIS days, and shading the energy-weighted average over the nine patterns. Nominal = model used for training; mismatched = perturbed plant (severity 1).

| Controller | Shading, nominal | Shading, mismatched | GMPP success, nominal (%) | GMPP success, mismatched (%) | Standard scenarios, nominal | Standard scenarios, mismatched | Real days, nominal | Real days, mismatched |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P&O | 80.41 | 81.10 | 33.3 | 33.3 | 99.73 | 99.75 | 99.80 | 99.82 |
| IncCond | 80.44 | 81.15 | 33.3 | 33.3 | 99.77 | 99.77 | 99.81 | 99.82 |
| Poly2(G,T)-MPPT | 78.17 | 77.20 | 33.3 | 22.2 | 99.81 | 99.34 | 99.96 | 99.25 |
| ANN-MPPT | 77.99 | 76.98 | 33.3 | 22.2 | 100.00 | 99.64 | 100.00 | 99.43 |
| Scan+P&O (no ML) | 99.36 | 99.36 | 100 | 100 | 99.58 | 99.61 | 99.64 | 99.68 |
| DNN scan-only (no P&O) | 99.30 | 99.33 | 100 | 100 | 99.58 | 99.57 | 99.73 | 99.71 |
| Hybrid (DNN+P&O) | 99.32 | 99.37 | 100 | 100 | 99.57 | 99.61 | 99.65 | 99.68 |
| Tree scan+P&O | 99.37 | 99.39 | 100 | 100 | 99.58 | 99.61 | 99.65 | 99.68 |
| PSO | 95.68 | 95.88 | 100 | 100 | 97.96 | 98.09 | 97.15 | 97.58 |

&#91;image: Fig. 21\]

**Fig. 13** Time response over the nine partial-shading patterns of 400 control cycles each (nominal plant): power with the true maximum power point (dashed) and the 98–100% band, PV voltage, and duty cycle (display only, D = 1 − V\_cmd/V\_bus); the left column shows the full run and the right column a zoom; triangles mark the scan cycles of the scan-based controllers.

### Uniform and variable irradiance

Without shading the ranking changes (Table 12). On the nominal plant the plain ANN is best (100.00% on the four standard scenarios and on the two real days), followed by Poly2(G,T) (99.81% and 99.96%), IncCond (99.77% and 99.81%) and P&O (99.73% and 99.80%). The ANN and Poly2 are maps fitted to the same plant model on which they are tested, so these values are an upper bound and not evidence of what they would achieve on hardware (see the mismatch results below). The scan-based controllers reach 99.57–99.58% on the standard scenarios, about 0.15 points below P&O, which is almost exactly their scan overhead (0.15% of the energy in these scenarios). On the smooth real days all scan-based controllers exceed 99.6% (**Figs. 14 and 15**). PSO is the weakest controller here (97.96% and 97.15%) because every restart costs several tens of cycles.

**Table 12** Tracking efficiency (%) of all controllers in each non-shading scenario (nominal plant). Step: irradiance step 400 → 800 W/m²; cloud: cloud passing; ramp: irradiance ramp; comprehensive: cloud-like irradiance variation with temperature drift; clear and variable day: two held-out PVGIS days (24 June 2020 and 6 April 2020), time-compressed to 3,600 control cycles.

| Controller | Step | Cloud | Ramp | Comprehensive | Clear day | Variable day |
| --- | --- | --- | --- | --- | --- | --- |
| P&O | 99.81 | 99.77 | 99.81 | 99.54 | 99.82 | 99.77 |
| IncCond | 99.89 | 99.78 | 99.83 | 99.57 | 99.83 | 99.78 |
| Poly2(G,T)-MPPT | 99.99 | 99.74 | 99.99 | 99.51 | 99.95 | 99.97 |
| ANN-MPPT | 100.00 | 99.99 | 100.00 | 99.99 | 100.00 | 100.00 |
| Scan+P&O (no ML) | 99.67 | 99.62 | 99.66 | 99.35 | 99.66 | 99.63 |
| DNN scan-only (no P&O) | 99.80 | 99.17 | 99.77 | 99.56 | 99.69 | 99.76 |
| Hybrid (DNN+P&O) | 99.61 | 99.63 | 99.65 | 99.40 | 99.66 | 99.63 |
| Tree scan+P&O | 99.65 | 99.63 | 99.66 | 99.38 | 99.66 | 99.64 |
| PSO | 99.16 | 96.19 | 98.81 | 97.66 | 97.83 | 96.47 |

&#91;image: Fig. 23\]

**Fig. 14** Irradiance step from 400 to 800 W/m² (nominal plant): power with the true maximum power point (dashed) and the 98–100% band, PV voltage and duty cycle (display only); left, full run; right, zoom around the step.

&#91;image: Fig. 25\]

**Fig. 15** Real clear held-out day (nominal plant); layout as in Fig. 14.

### Plant mismatch

The plain ANN and Poly2(G,T) were fitted to the nominal plant, and their accuracy degrades as the evaluated plant departs from it (Table 13). On the comprehensive scenario the ANN falls from 99.99% to 98.99% and Poly2 from 99.51% to 97.85% as the severity rises from 0 to 2. The ANN crosses below P&O between severity 1 and 1.5, and Poly2 is already below it at severity 0.5. P&O (99.54–99.57%) and the hybrid (99.40–99.43%) are flat, but the hybrid remains about 0.1–0.15 points below P&O throughout. On the standard scenarios at severity 1, P&O and IncCond (99.75% and 99.77%) still lead the hybrid (99.61%) and the ANN (99.64%); on the two real days the hybrid (99.68%) is ahead of the ANN (99.43%) and Poly2 (99.25%) but behind P&O (99.82%). Under partial shading the hybrid stays at 99.3–99.5% at every severity (99.32%, 99.37%, 99.47% at severities 0, 1 and 2), while P&O remains near 80–82% and the ANN near 76–78%. Severities 1.5 and 2 lie outside the domain-randomization range of the hybrid DNN, and the scan-based results do not degrade there because the final tracking is done on the measured plant.

**Table 13** Tracking efficiency (%) against plant-mismatch severity (0 = nominal plant, 1 = headline mismatched plant, 2 = largest severity) for the comprehensive scenario and for partial shading (nine patterns).

| Controller | Comprehensive, s = 0 | Comprehensive, s = 1 | Comprehensive, s = 2 | Shading, s = 0 | Shading, s = 1 | Shading, s = 2 |
| --- | --- | --- | --- | --- | --- | --- |
| P&O | 99.54 | 99.56 | 99.56 | 80.41 | 81.10 | 81.58 |
| IncCond | 99.57 | 99.60 | 99.61 | 80.44 | 81.15 | 81.61 |
| Poly2(G,T)-MPPT | 99.51 | 98.96 | 97.85 | 78.17 | 77.20 | 76.26 |
| ANN-MPPT | 99.99 | 99.76 | 98.99 | 77.99 | 76.98 | 76.03 |
| Scan+P&O (no ML) | 99.35 | 99.43 | 99.52 | 99.36 | 99.36 | 99.45 |
| DNN scan-only (no P&O) | 99.56 | 99.60 | 99.51 | 99.30 | 99.33 | 99.22 |
| Hybrid (DNN+P&O) | 99.40 | 99.42 | 99.43 | 99.32 | 99.37 | 99.47 |
| Tree scan+P&O | 99.38 | 99.39 | 99.45 | 99.37 | 99.39 | 99.44 |
| PSO | 97.66 | 97.92 | 98.14 | 95.68 | 95.88 | 96.12 |

### Cold start, convergence and oscillation

From a cold start (five initial voltages between 13.4 and 28.9 V), the four slew-limited controllers need on average 9.4 (P&O), 8.6 (IncCond) and 7.8 (Poly2 and ANN) cycles to reach 98% of the maximum, simply because they must travel the distance at 0.5 V per cycle (**Fig. 16**, Table 14). Scan+P&O needs 4.0 cycles because its estimate lies on a 2 V probe grid and the P&O stage must then walk to the peak. The DNN-based scan controllers and the tree-based variant need 1.0 cycle because their estimate is precise. PSO needs 22.8 cycles. This advantage assumes that the converter can perform the sweep and the hand-off jump without the 0.5 V per cycle limit. The converter model of Table 7 supports the timing part of this assumption: a 14-point scan needs at most 14 × 60 ms = 0.84 s, which fits inside one 1 s control cycle, but the supervisory sequencing and the sensor sampling during the sweep were not modeled. When the starting points are extended to 15, 20, 26, 30 and 35 V (the last two close to or above the open-circuit voltage of 32.9 V), P&O and IncCond do not converge within the episode from two of the five starts, whereas all other controllers do. After events in the standard scenarios, convergence takes 0.9–2.9 cycles for all controllers except PSO (29.6 cycles).

P&O and IncCond oscillate by about 1 V peak to peak (1.16 and 1.04 V), and so do the scan-based controllers whose second stage is P&O (1.17 V for Scan+P&O, 1.23 V for the hybrid and 1.16 V for the tree-based variant). The model-based controllers (Poly2 0.04 V, ANN 0.03 V) and the DNN scan-only variant (0.00 V) are nearly ripple-free. The hybrid does not remove the P&O oscillation; it inherits it. PSO has the largest oscillation (2.11 V). The corresponding power oscillation is below 2% for every controller except PSO (mean 8%, up to 15%), so for P&O-based controllers the ripple costs little energy on this flat P–V peak (**Table 14**).

**Table 14** Dynamic behavior and cost of the controllers on the quasi-static nominal plant. Event convergence: cycles until P/P\_mpp ≥ 98% for 10 consecutive cycles after an irradiance step or a shading change (standard scenarios and shading suite). Cold start: mean cycles from five valid initial voltages; the second cold-start column counts the starts that never converged when the initial voltage is extended to 30 and 35 V. Decision time: mean controller computation per cycle, plant model excluded (Python/numpy on the machine of Table 15). Scan overhead: share of the energy lost to scan cycles in the standard scenarios.

| Controller | Event convergence (cycles) | Oscillation (V p-p) | Cold start (cycles) | Not converged, extended starts | Decision time (µs/cycle) | Scan overhead (%) |
| --- | --- | --- | --- | --- | --- | --- |
| P&O | 1.75 | 1.16 | 9.4 | 2/5 | 8.25 | 0 |
| IncCond | 2.50 | 1.04 | 8.6 | 2/5 | 9.08 | 0 |
| Poly2(G,T)-MPPT | 0.88 | 0.04 | 7.8 | 0/5 | 42.2 | 0 |
| ANN-MPPT | 0.88 | 0.03 | 7.8 | 0/5 | 55.8 | 0 |
| Scan+P&O (no ML) | 2.88 | 1.17 | 4.0 | 0/5 | 9.71 | 0.15 |
| DNN scan-only (no P&O) | 1.75 | 0.00 | 1.0 | 0/5 | 0.52 | 0.15 |
| Hybrid (DNN+P&O) | 1.75 | 1.23 | 1.0 | 0/5 | 8.90 | 0.15 |
| Tree scan+P&O | 1.75 | 1.16 | 1.0 | 0/5 | 33.3 | 0.15 |
| PSO | 29.62 | 2.11 | 22.8 | 0/5 | 5.38 | 0 |

&#91;image: Fig. 29\]

**Fig. 16** Cold start from the lowest (left) and the highest (right) of five valid initial voltages (V0 = 13.4 V and 28.9 V; G = 800 W/m², T = 25 °C); the scan cycles show the mean scan power. The slew-limited controllers need up to 19 (P&O) and 18 (ANN) cycles from the lowest start.

### Robustness studies

Four further studies test whether the conclusions depend on the seed, the estimator design, the plant model or the measurement chain; their tables and figure are given in the Supplementary material (Tables S1–S7, Fig. S1). Over 10 scenario seeds, with the stage-1 DNN retrained with five seeds, the hybrid beats P&O under partial shading by 18.9 points (p = 0.002, ten of ten seeds), whereas the hybrid and Scan+P&O do not differ significantly under shading on the nominal plant (p = 0.23); the DNN scan-only variant is less reliable than the hybrid (98.32 ± 1.00% against 99.36 ± 0.05%) (Tables S1 and S2). Replacing the stage-1 network by smaller or larger networks, regularized variants or a tree-based estimator leaves the closed-loop shading efficiency within 99.33–99.37%, a linear regressor reaches 97.79%, and a network without the scan currents falls to 80.18% (Table S3). On a plant that includes an averaged boost converter the efficiencies change by at most 0.58 points (the plain ANN under shading) and the ranking of the controllers is unchanged; the averaged and the switching-level converter models deviate by at most 0.049 V outside the large scan jumps (Tables S4 and S5, Fig. S1). With sensor noise (0.5% on current and voltage, 2% on irradiance) the scan-based controllers lose about 0.6 points but still solve every pattern (Table S6); half-precision storage leaves the estimate unchanged, whereas naive int8 quantization raises the RMSE from 0.78 to 1.98 V (Table S7).

### Computation and parallel computing

The controller decision time per control cycle (Table 14) is 8.3 µs for P&O, 9.1 µs for IncCond, 42 µs for Poly2(G,T) and 56 µs for the plain ANN, which evaluates a network every cycle, against 8.9 µs for the hybrid, which calls its network only at scans (0.42% of the cycles, 0.19% of the available energy; the scan-only variant averages 0.5 µs). The network itself takes a median of 19 µs (99th percentile 87 µs) per inference in numpy float32, which is 2 × 10⁻⁵ of a 1 s control cycle; a call through a deep-learning framework is slower, with a median of 2.9 ms for a direct model call and 65 ms for a high-level predict() call. These are Python timings on a general-purpose CPU, not microcontroller measurements; they show that the online cost of the DNN is negligible against the control period, and that the relevant embedded questions are memory (108 kB in float32, 55 kB in float16) and the sensing hardware needed for the scan.

Parallel computing matters elsewhere in the pipeline (Table 15, **Fig. 17**). Generating the 200,000 domain-randomized training plants took 37 s on four worker processes, and the parallel result is bit-identical to the serial one; the speed-up is 1.62× on two workers and 2.62× on four (efficiency 0.65), and 1.90× with eight workers on the four available CPUs (oversubscribed). Seed sweeps of the controller suites scale better (1.94× on two and 3.82× on four workers, efficiency 0.95) because their tasks are longer. One simulated Hybrid run, including the plant model, proceeds at 377 control cycles per second on one core, so the 10-seed, 9-controller, two-plant suites of this section, the ablation and the noise and converter studies are the workload for which a cluster or multi-core machine is useful. The online controller is not.

**Table 15** Where parallel computing and where real-time inference matter. The numbers were measured on the machine recorded with the results (4 logical CPUs). Int8 and float16 timings are numpy emulations and not microcontroller measurements; worker counts above the CPU count are oversubscribed.

| Part | Role | Measure | Result |
| --- | --- | --- | --- |
| Training-data generation (200,000 plants) | Parallel computing | Speed-up / efficiency at 2, 4, 8 workers | 1.62× / 0.81; 2.62× / 0.65; 1.90× / 0.24 |
| Seed sweeps and design studies (shading suites) | Parallel computing | Speed-up / efficiency at 2, 4, 8 workers | 1.94× / 0.97; 3.82× / 0.95; 3.76× / 0.47 |
| Simulation throughput (Hybrid, one core) | Parallel computing | Control cycles per second including the plant model | 377 |
| Online DNN inference, float32 numpy | Real time | Median / 99th percentile latency | 19 / 87 µs (2 × 10⁻⁵ of the cycle) |
| Online DNN inference, int8 numpy emulation | Real time | Median / 99th percentile latency | 69 / 191 µs |
| Online DNN inference, Keras model call | Real time | Median / 99th percentile latency | 2.9 / 5.3 ms |
| Online DNN inference, Keras predict() | Real time | Median / 99th percentile latency | 64.8 / 101 ms |
| Scan overhead (Hybrid, shading suite) | Real time | Share of cycles / share of available energy | 0.42% / 0.19% |

&#91;image: Fig. 38\]

**Fig. 17** (a) Parallel speed-up of the training-data generation and of the scenario-seed sweep; (b) DNN inference latency per precision and runtime against the 1 s control cycle.

### Summary of the comparison

Table 16 combines the design properties and the measured results of the controllers. Four conclusions follow from the evidence of this section. First, a global search by scanning is what solves partial shading: Scan+P&O reaches 99.36% under shading with no learning, against 80.4% for P&O. Second, the DNN does not add measurable robustness to this scan-based structure; replacing it by a gradient-boosted tree or a much smaller network gives the same closed-loop efficiency, and the benefit comes from the scan currents. Third, the DNN gives a faster cold start (1.0 cycle against 4.0 for Scan+P&O and 9.4 for P&O) under the stated assumption that the sweep and hand-off are not slew-limited, and it removes the 2 V granularity of the best-probe rule at a negligible online cost. Fourth, these gains come at the price of a small efficiency loss on uniform irradiance (0.15 points below P&O, equal to the scan overhead), the P&O oscillation of about 1.2 V that the hybrid inherits, and sensitivity to current-sensor noise, which reduces the shading efficiency by about 0.6 points.

**Table 16** Conceptual comparison of the controllers. The design columns are properties of the implementations in this study; the numeric columns are measured in this study (shading efficiency and standard-scenario efficiency in %, nominal / mismatched plant). The hybrid P&O method of the literature was not implemented.

| Controller | Inputs | Training need | Global search under shading | Shading eff. | GMPP success (nominal) | Standard scenarios eff. | Decision time (µs) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P&O | V, I | None | No (local) | 80.41 / 81.10 | 33.3% | 99.73 / 99.75 | 8.3 |
| IncCond | V, I | None | No (local) | 80.44 / 81.15 | 33.3% | 99.77 / 99.77 | 9.1 |
| Hybrid P&O (literature) | Not implemented here | Not implemented here | Not implemented here | – | – | – | – |
| Poly2(G,T)-MPPT | G, T | Offline, plant data | No (one V\_ref) | 78.17 / 77.20 | 33.3% | 99.81 / 99.34 | 42.2 |
| ANN-MPPT | G, T, V, I | Offline, simulated or measured data | No (one V\_ref) | 77.99 / 76.98 | 33.3% | 100.00 / 99.64 | 55.8 |
| Scan+P&O (no ML) | V, I, I–V scan (14 probes) | None | Yes (scan, argmax, local P&O) | 99.36 / 99.36 | 100% | 99.58 / 99.61 | 9.7 |
| DNN scan-only (no P&O) | Scan, G, T | Offline, domain-randomized plants | Yes (DNN picks V\_gmpp) | 99.30 / 99.33 | 100% | 99.58 / 99.57 | 0.5 |
| Hybrid (DNN+P&O) | Scan, G, T | Offline, domain-randomized plants | Yes (scan, DNN hand-off, local P&O) | 99.32 / 99.37 | 100% | 99.57 / 99.61 | 8.9 |
| Tree scan+P&O | Scan, G, T | Offline, domain-randomized plants | Yes (scan, tree hand-off, local P&O) | 99.37 / 99.39 | 100% | 99.58 / 99.61 | 33.3 |
| PSO | V, I | None | Yes (swarm, slow) | 95.68 / 95.88 | 100% | 97.96 / 98.09 | 5.4 |

## Discussion

### What the DNN contributes and what it does not

The central result is that the architecture, not the network, makes the controller robust to partial shading. A scan followed by P&O reaches the global maximum in all nine patterns with or without a DNN (99.36% for Scan+P&O against 99.32% for the hybrid, a difference that is not significant over 10 seeds, p = 0.23), and the same holds on the mismatched plant (99.36% against 99.37%) and on the converter plant (99.36% against 99.32%). The ablation goes further: a gradient-boosted-tree estimator (99.37%), a network with 1,633 parameters (99.35%) and a network with 103,169 parameters (99.33%) give the same closed-loop efficiency, and only the scan currents matter: without them the same network falls to 80.18%, the level of P&O. The DNN adds three measurable but limited benefits. Its first estimate is more accurate than the best probe on uniform irradiance (99.97% against 99.50% of the available power); it reduces the cold-start convergence from 4.0 to 1.0 cycle, because the estimate is precise rather than on a 2 V probe grid; and the DNN scan-only variant has no steady-state ripple, although it is less reliable than the hybrid across seeds (98.32 ± 1.00% against 99.36 ± 0.05% under shading). In closed loop, however, P&O closes the accuracy gap within a few cycles, so the efficiency gain is negligible. Outside the training range the benefit reverses: the best-probe rule is the better estimator (99.37% against 98.44%), and the conformal coverage falls from 0.90 to 0.41. A network can therefore trade robustness for precision, and its value depends on how closely the deployed plant resembles the simulated training family.

These findings do not show that neural networks are unsuitable for MPPT. They show that a reported gain over P&O alone is not evidence for the network unless the same architecture without the network is also tested. The magnitude of the gain reported here under shading, about 19 points over P&O, is of the same order as the 15.6–49% reported for a hybrid with a small multilayer perceptron [19], but in the present setting it is obtained without the network. Whether networks that avoid a scan, or that learn shading directly from voltage and current, can match a scan-based baseline remains an open question.

### Why the plain ANN is best on the nominal plant and where it fails

The plain ANN reaches 100.00% on the standard scenarios of the nominal plant because its labels and the closed-loop plant derive from the same physics model, so the in-distribution score is optimistic and should be read as an upper bound. Its weaknesses appear as soon as that assumption is relaxed. It cannot see shading, since irradiance and temperature do not describe which substring is shaded, so it stays on a local peak (78.0%). It degrades with plant mismatch, from 99.99% to 98.99% over the sweep, and crosses below P&O at a severity between 1 and 1.5. The fair equal-information comparison is Poly2(G,T), which degrades faster (99.51% to 97.85%). Finally, it evaluates a network at every cycle and takes about 56 µs per decision, roughly seven times the decision time of P&O (8.3 µs).

### Gaps that are not closed

On uniform irradiance the hybrid is about 0.15 points below P&O, almost exactly its scan overhead, and about 0.4 points below the plain ANN on the nominal plant. On the mismatched plant, P&O and IncCond still lead the hybrid by about 0.14 points on the standard scenarios. The P&O stage of the hybrid adds little on this flat P–V peak: the DNN scan-only variant reaches the same efficiency with zero ripple, and the second stage contributes mainly adaptation to mismatch (about 0.04 points) and robustness when irradiance drifts between scans, and it protects against occasional wrong-peak estimates (the scan-only variant reaches 98.32 ± 1.00% across seeds under shading, the hybrid 99.36 ± 0.05%). Conformal intervals are valid in distribution only. The hybrid inherits the P&O oscillation of about 1.2 V; it loses about 0.6 points when the current and voltage sensors carry 0.5% noise and the irradiance input 2% (the scan-only variant is unaffected); and a naive post-training int8 quantization raises the estimation error from 0.78 to 1.98 V. These limits define the regime in which the hybrid is worthwhile: installations where partial shading is frequent, where the plant differs from its model, and where a scan can be performed cheaply.

### Practical and hardware considerations

The proposed controllers need more than P&O. They require an irradiance and a temperature input, because the DNN uses them and the re-scan rule is based on P/G, and a converter that can sweep about 14 operating points quickly. The simulation treats the scan as one control cycle at the mean probe power; on real converters a sweep takes milliseconds to hundreds of milliseconds, which would raise the scan overhead above 0.15%, and the cold-start result of 1.0 cycle assumes the sweep and the handoff jump are not limited by the 0.5 V per cycle actuator limit. The DNN is small: its 27,009 parameters correspond to about 108 kB in 32-bit floating point and roughly 27 thousand multiply–accumulate operations per inference, and it runs only at scans, at most once every 30 cycles. This is compatible with microcontroller-class hardware, but it was measured here only on a CPU in Python (a median of 19 µs per inference in numpy and 2.9 ms through a Keras call; the float16 and int8 timings are numpy emulations) and was not tested on embedded hardware or in hardware-in-the-loop operation.

### Relevance to high-performance computing

The computational burden of this approach lies in the offline pipeline, not in deployment, and the measurements separate the two. Generating the 200,000 randomized plant–shading samples took 37 s on four worker processes; the speed-up over serial execution is 1.62× on two, 2.62× on four (efficiency 0.65) and 1.90× on eight workers on the four available CPUs (oversubscribed), and the result is bit-identical to the serial one. The seed sweeps, ablations and noise and converter studies, which make the comparison statistically credible, scale better (3.82× on four workers, efficiency 0.95) because their tasks are longer; one simulated hybrid run proceeds at only 377 control cycles per second, so a full evaluation with ten seeds, nine controllers and two plants is a workload for which multi-core or cluster resources are worthwhile. Parallel and high-performance resources become more important when the randomization space is enlarged, for example to longer strings with more substrings, finer shading patterns, additional plant parameters and sensor noise, or when a separate network is trained for each plant in a large fleet. The online controller is a different matter: the network needs a median of 19 µs per inference (99th percentile 87 µs) against a 1 s control period and runs only at scans (0.42% of the cycles), so deployment calls for real-time determinism on embedded hardware rather than for high-performance computing. Whether the scan, the sensing and the hand-off meet the timing of a real converter is the open real-time question, which a simulation study cannot answer.

## Conclusion

This study separated the contribution of a deep neural network from that of the scan strategy in a hybrid MPPT controller under partial shading. A two-stage controller, a fast I–V scan followed by P&O, was evaluated with and without a DNN, using a corrected single-diode model, 16 years of hourly PVGIS-SARAH3 data for Ramadi and a DNN trained on 200,000 domain-randomized plants with shading patterns.

The scan-based controllers reached the global maximum in all nine shading patterns with 99.3–99.4% tracking efficiency, against 80.4% for P&O and 78.0% for a plain ANN-MPPT, which reached it in only three. The same performance was obtained without a neural network (99.36% for Scan+P&O against 99.32% for the hybrid; the paired difference is not significant over 10 seeds), with a gradient-boosted-tree estimator (99.37%) and with networks of 1,633 to 103,169 parameters, so the robustness to shading comes from the scan architecture and the information in the scan currents. The DNN improved the precision of the first estimate and the cold-start convergence (1.0 against 4.0 cycles), but outside its training range the simple best-probe rule was the better estimator (99.37% against 98.44%) and the conformal coverage fell from 0.90 to 0.41. The hybrid also remained about 0.15 points below P&O under uniform irradiance, which equals its scan overhead of 0.15%, kept the P&O oscillation of about 1.2 V, and lost about 0.6 points under sensor noise. Its decision time per control cycle (8.9 µs) was similar to that of P&O (8.3 µs) and about six times lower than that of a plain ANN-MPPT (56 µs), because the network runs only at scans. Parallel computing accelerated the offline pipeline (2.62× on four workers for data generation and 3.82× for seed sweeps) but is not needed by the online controller.

The main limitations are the following. (1) All results are simulations; no hardware or hardware-in-the-loop test was performed, and the converter model, although verified against a switching-level model, was not compared with a physical converter. (2) The training labels, the nominal plant and the evaluation plants derive from the same physics model, and the mismatched plant is a declared parameter perturbation. (3) The scan was modeled as one control cycle at the mean probe power; the converter model shows that a 14-point sweep fits in 0.84 s, but supervisory sequencing and sensor sampling were not modeled, and the cold-start advantage depends on this assumption. (4) The proposed controllers require irradiance and temperature inputs and a fast I–V scan capability, which adds hardware cost compared with sensor-less P&O. (5) One module type, one site and one three-substring configuration were studied, the nine shading patterns belong to the same family as the training patterns, and the scan and re-scan constants were fixed a priori rather than optimized. (6) The conformal intervals are valid only inside the training distribution. (7) Computation times refer to a Python implementation on a CPU and vary with the machine; the float16 and int8 timings are numpy emulations, and a naive int8 quantization degraded the estimate. (8) The comparison with other published methods is qualitative (Table 1): apart from the standard algorithms and the PSO comparator described here, no published method was re-implemented.

Future work should test the controller in hardware-in-the-loop and on a prototype with realistic scan durations and sensor noise, add an out-of-distribution detector that falls back to the best-probe rule, extend the randomization to longer strings and richer shading patterns, and evaluate whether networks that avoid the scan can match a scan-based baseline under the same ablation protocol.

## Declarations

**Funding.** The authors received no funding for this work.

**Competing interests.** The authors declare that they have no competing interests.

**Ethics approval and consent to participate.** Not applicable.

**Data availability.** The irradiance and temperature data are from the public PVGIS-SARAH3 database of the European Commission Joint Research Centre [20], for the location 33.42° N, 43.30° E, 2005–2020. The hourly series used in this study and all result tables are available in the project repository.

**Code availability.** The simulation code (controllers, plant and converter models, data generation, training and evaluation scripts), the trained models, the data files behind every figure and all result tables, with a changelog of the numbers, are available at https://github.com/engw7/Waleed-Hameed-s-Paper-work-s-Files (release v1.0.0) and are archived on Zenodo (https://doi.org/10.5281/zenodo.23706406).

**Author contributions.** W.H.: conceptualization, methodology, software, validation, formal analysis, investigation, data curation, writing—original draft, visualization. K.D.: supervision, writing—review and editing.

**Use of AI tools.** The authors used an AI assistant (Claude, Anthropic) to support code development, debugging and drafting. The authors reviewed and verified all code, results and text and take full responsibility for the content.

## References

[1] Krishnaram, K., Padmanabhan, T.S., Alsaif, F., Senthilkumar, S.: Performance optimization of interleaved boost converter with ANN supported adaptable stepped-scaled P&O based MPPT for solar powered applications. Sci. Rep. 14 (2024). https://doi.org/10.1038/s41598-024-58852-8

[2] Endiz, M.S., Gökkuş, G., Coşgun, A.E., Demir, H.: A review of traditional and advanced MPPT approaches for PV systems under uniformly insolation and partially shaded conditions. Appl. Sci. 15(3), 1031 (2025). https://doi.org/10.3390/app15031031

[3] Abidi, H., Sidhom, L., Chihi, I.: Systematic literature review and benchmarking for photovoltaic MPPT techniques. Energies 16(8), 3509 (2023). https://doi.org/10.3390/en16083509

[4] Sørensen, R., Mihet-Popa, L.: Comparative evaluation of traditional and advanced algorithms for photovoltaic systems in partial shading conditions. Solar 4(4), 572–594 (2024). https://doi.org/10.3390/solar4040027

[5] Eleraky, H.G., Refaat, A., Kalas, A., Abouzeid, A.F.: Performance assessment of artificial neural networks-based MPPT technique for photovoltaic systems. In: ICEENG 2024 – 14th IEEE International Conference on Electrical Engineering, pp. 48–53. IEEE (2024). https://doi.org/10.1109/ICEENG58856.2024.10566372

[6] Devarakonda, A.K., Karuppiah, N., Selvaraj, T., Balachandran, P.K., Shanmugasundaram, R., Senjyu, T.: A comparative analysis of maximum power point techniques for solar photovoltaic systems. Energies 15(22), 8776 (2022). https://doi.org/10.3390/en15228776

[7] Mishra, P.K., Tiwari, P.: Incremental conductance MPPT in grid connected PV system. Int. J. Eng. Sci. Technol. 13(1), 138–145 (2021). https://doi.org/10.4314/ijest.v13i1.21s

[8] Abouzeid, A.F., Eleraky, H., Kalas, A., Rizk, R., Elsakka, M.M., Refaat, A.: Experimental validation of a low-cost maximum power point tracking technique based on artificial neural network for photovoltaic systems. Sci. Rep. 14 (2024). https://doi.org/10.1038/s41598-024-67306-0

[9] Huang, Y., Wang, H., Cao, G., Lu, T., Wang, C., Li, H.: Research on photovoltaic MPPT algorithm based on wolf pack algorithm. In: 2021 China Automation Congress (CAC), pp. 2942–2946. IEEE (2021). https://doi.org/10.1109/CAC53003.2021.9727757

[10] Shyam, S., Ghosh, A.: Study on artificial neural network based MPPT algorithm in PV application. In: 2023 IEEE 3rd International Conference on Sustainable Energy and Future Electric Transportation (SeFeT). IEEE (2023). https://doi.org/10.1109/SeFeT57834.2023.10245966

[11] Yishu, Z., Yilun, L., Leiyu, C., He, G., Fuguang, L., Yinghe, Z.: Application of composite control algorithm in photovoltaic MPPT. In: 2022 Chinese Automation Congress (CAC), pp. 1314–1319. IEEE (2022). https://doi.org/10.1109/CAC57257.2022.10055576

[12] Villegas-Mier, C.G., Rodriguez-Resendiz, J., Álvarez-Alvarado, J.M., Rodriguez-Resendiz, H., Herrera-Navarro, A.M., Rodríguez-Abreo, O.: Artificial neural networks in MPPT algorithms for optimization of photovoltaic power systems: a review. Micromachines 12(10), 1260 (2021). https://doi.org/10.3390/mi12101260

[13] Gugulothu, A., Naik, K.A.: Integrating artificial neural networks into hybrid perturb and observe algorithm for improved photovoltaic system performance. In: 2023 3rd Asian Conference on Innovation in Technology (ASIANCON). IEEE (2023). https://doi.org/10.1109/ASIANCON58793.2023.10270268

[14] Pradeep Kumar, S., Viswambharan, V.K., Pillai, S.: Performance analysis of maximum power point tracking of PV systems using artificial neural networks and support vector machines. In: 2023 3rd International Conference on Computational Intelligence and Knowledge Economy (ICCIKE), pp. 511–515. IEEE (2023). https://doi.org/10.1109/ICCIKE58312.2023.10131783

[15] Nour Ali, M.: Improved design of artificial neural network for MPPT of grid-connected PV systems. In: 2018 20th International Middle East Power Systems Conference (MEPCON), pp. 97–102. IEEE (2018). https://doi.org/10.1109/MEPCON.2018.8635202

[16] Bouchetob, E., Nadji, B.: Using ANN based MPPT controller to increase PV central performance. In: 2022 2nd International Conference on Advanced Electrical Engineering (ICAEE). IEEE (2022). https://doi.org/10.1109/ICAEE53772.2022.9961982

[17] Kamrul Islam, O., Ahmed, M.S., Rahman, K., Tahsin, T.: A comprehensive comparison between boost and buck-boost converters in solar MPPT with ANN. In: 2020 Emerging Technology in Computing, Communication and Electronics (ETCCE). IEEE (2020). https://doi.org/10.1109/ETCCE51779.2020.9350867

[18] Dawahdeh, A., Sharadga, H., Kumar, S.: Novel MPPT controller augmented with neural network for use with photovoltaic systems experiencing rapid solar radiation changes. Sustainability 16(3), 1021 (2024). https://doi.org/10.3390/su16031021

[19] Cruz, B., Ricalde, L., Quintal-Palomo, R., Bassam, A., Rico-Camacho, R.I.: Hybrid artificial neural network and perturb & observe strategy for adaptive maximum power point tracking in partially shaded photovoltaic systems. Energies 18(19), 5053 (2025). https://doi.org/10.3390/en18195053

[20] Huld, T., Müller, R., Gambardella, A.: A new solar radiation database for estimating PV performance in Europe and Africa. Sol. Energy 86(6), 1803–1815 (2012). https://doi.org/10.1016/j.solener.2012.03.006

[21] Wang, K., Hong, D., Ma, J., Man, K.L., Huang, K., Huang, X.: Maximum power point tracking of photovoltaic systems using deep Q-networks. In: 2020 IEEE International Conference on Industrial Informatics (INDIN), pp. 100–103. IEEE (2020). https://doi.org/10.1109/INDIN45582.2020.9442100

[22] Ortiz-Muñoz, D., Luviano-Cruz, D., Perez-Dominguez, L.A., Rodriguez-Ramirez, A.G., Garcia-Luna, F.: Hybrid fuzzy–DDPG approach for efficient MPPT in partially shaded photovoltaic panels. Appl. Sci. 15(9), 4869 (2025). https://doi.org/10.3390/app15094869

[23] De Soto, W., Klein, S.A., Beckman, W.A.: Improvement and validation of a model for photovoltaic array performance. Sol. Energy 80(1), 78–88 (2006). https://doi.org/10.1016/j.solener.2005.06.010

[24] Duffie, J.A., Beckman, W.A.: Solar Engineering of Thermal Processes, 4th edn. Wiley, Hoboken (2013)

[25] Jain, A., Kapoor, A.: Exact analytical solutions of the parameters of real solar cells using Lambert W-function. Sol. Energy Mater. Sol. Cells 81(2), 269–277 (2004). https://doi.org/10.1016/j.solmat.2003.11.018

[26] Patel, H., Agarwal, V.: MATLAB-based modeling to study the effects of partial shading on PV array characteristics. IEEE Trans. Energy Convers. 23(1), 302–310 (2008). https://doi.org/10.1109/TEC.2007.914308

[27] Kingma, D.P., Ba, J.: Adam: a method for stochastic optimization. In: 3rd International Conference on Learning Representations (ICLR 2015), San Diego, CA, USA (2015). arXiv:1412.6980

[28] Lei, J., G’Sell, M., Rinaldo, A., Tibshirani, R.J., Wasserman, L.: Distribution-free predictive inference for regression. J. Am. Stat. Assoc. 113(523), 1094–1111 (2018). https://doi.org/10.1080/01621459.2017.1307116
