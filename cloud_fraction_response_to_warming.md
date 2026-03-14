# Cloud Fraction Response to Warming: A Thermodynamic Constraint Framework

**Authors:** Hassan Beydoun, Adam Varble, Mohammad Fathi

**Date:** March 2026

**Status:** Working Document - For Internal Review

---

## Abstract

This document presents a theoretical framework explaining the surprisingly weak response of tropical cloud fraction to surface warming. While established thermodynamic constraints predict a ~5%/K decrease in upward mass flux on isotherms, observations from Global Storm-Resolving Model (GSRM) simulations reveal that cloud fraction decreases by only ~1%/K. We demonstrate that this apparent discrepancy arises from compensating effects: the decrease in atmospheric density at constant temperature partially offsets the mass flux reduction, while cloud microphysical timescales exhibit minimal change. The cloud fraction response can be understood as the ratio of subsidence timescale to cloud dissipation timescale, both of which are modulated by changes in pressure thickness on isotherms.

---

## 1. Introduction

Understanding how cloud fraction responds to global warming remains one of the central challenges in climate science (Bony et al., 2015; Zelinka et al., 2020). Clouds exert a profound influence on Earth's radiation budget, and uncertainty in cloud feedbacks dominates the spread in equilibrium climate sensitivity estimates across climate models (Sherwood et al., 2020).

A fundamental constraint on tropical convection comes from the thermodynamic balance between radiative cooling and adiabatic warming through subsidence. On isotherms (surfaces of constant temperature), the upward mass flux is constrained by:

$$\omega = \frac{Q}{S}$$

where $\omega$ is the vertical velocity in pressure coordinates (Pa/s), $Q$ is the radiative heating rate (K/s), and $S$ is the static stability (K/Pa). This relationship, derived from the thermodynamic energy equation under weak temperature gradient assumptions, has been well-established in tropical dynamics (Held & Soden, 2006; Bony et al., 2013).

With warming, $Q$ remains approximately constant on isotherms while $S$ increases due to the Clausius-Clapeyron scaling of saturation specific humidity. This leads to a robust ~5%/K decrease in mass flux across climate models (Held & Soden, 2006; Vecchi & Soden, 2007).

The central question addressed here is: **Why does cloud fraction decrease by only ~1%/K on isotherms when the mass flux decreases by ~5%/K?**

---

## 2. Theoretical Framework

### 2.1 The Cloud Fraction Budget

We develop a steady-state framework for cloud fraction based on the balance between cloud sources and sinks. Following previous work on anvil cloud dynamics (Hartmann & Larson, 2002; Kuang & Hartmann, 2007; Beydoun et al., 2021), cloud fraction can be expressed as:

$$\text{CF} = \delta \cdot M \cdot \tau$$

where:
- $\delta$ is the fractional detrainment rate (m$^{-1}$ or Pa$^{-1}$)
- $M$ is the convective mass flux (kg m$^{-2}$ s$^{-1}$)
- $\tau$ is the cloud lifetime (s)

This formulation emerges from the steady-state balance:

$$\delta \cdot \omega \cdot q = \frac{\text{CF} \cdot q}{\tau}$$

where $q$ is the in-cloud condensate mixing ratio. The left-hand side represents the source of cloud condensate from detraining convective updrafts, while the right-hand side represents the sink from microphysical processes (sedimentation, evaporation/sublimation, precipitation formation).

### 2.2 Mass Balance Perspective

An alternative and illuminating perspective comes from the clear-sky/cloudy-sky mass balance. At anvil levels, mass continuity requires:

$$(1 - \text{CF}) \cdot \omega_{\text{down}} = \text{CF} \cdot \omega_{\text{up}}$$

where $\omega_{\text{down}}$ is the clear-sky subsidence rate and $\omega_{\text{up}}$ is the in-cloud ascent rate (or more precisely, the effective vertical motion associated with cloud maintenance).

Dividing both sides by the pressure thickness $\Delta p$ over which the cloud layer exists:

$$(1 - \text{CF}) \cdot \tau^{-1}_{\text{sub}} = \text{CF} \cdot \tau^{-1}_{\text{cloud}}$$

where we have defined inverse timescales:
- $\tau^{-1}_{\text{sub}} = \omega_{\text{down}} / \Delta p$ : the inverse subsidence timescale
- $\tau^{-1}_{\text{cloud}} = \omega_{\text{up}} / \Delta p$ : the inverse cloud processing timescale

Solving for cloud fraction:

$$\text{CF} = \frac{\tau^{-1}_{\text{sub}}}{\tau^{-1}_{\text{sub}} + \tau^{-1}_{\text{cloud}}}$$

Since $\tau_{\text{cloud}} \ll \tau_{\text{sub}}$ (i.e., cloud microphysical processes are much faster than large-scale subsidence), this simplifies to:

$$\boxed{\text{CF} \approx \frac{\tau^{-1}_{\text{sub}}}{\tau^{-1}_{\text{cloud}}} = \frac{\tau_{\text{cloud}}}{\tau_{\text{sub}}}}$$

**This is a key result: cloud fraction is governed by the ratio of cloud dissipation timescale to subsidence timescale.**

### 2.3 The Inverse Timescales

#### 2.3.1 Subsidence Timescale

The inverse subsidence timescale is:

$$\tau^{-1}_{\text{sub}} = \frac{\omega_{\text{sub}}}{\Delta p} = \frac{Q/S}{\Delta p}$$

On isotherms with warming:
- $\omega_{\text{sub}}$ decreases at ~5%/K (from the $Q/S$ constraint)
- $\Delta p$ decreases at ~3%/K (due to lower density at constant temperature at higher altitude)

Therefore, $\tau^{-1}_{\text{sub}}$ decreases at only **~2%/K** rather than 5%/K.

#### 2.3.2 Cloud Dissipation Timescale

The cloud dissipation timescale is determined by in-cloud microphysical processes:

$$\tau^{-1}_{\text{cloud}} = \frac{\text{Sinks}}{\Delta q}$$

where:
- Sinks = sedimentation + evaporation/sublimation + autoconversion + accretion + melting (s$^{-1}$)
- $\Delta q$ = the condensate mixing ratio detrained from updrafts (kg/kg)

On isotherms with warming, both the numerator (Sinks) and denominator ($\Delta q$) increase due to:
- Higher absolute humidity at constant temperature (Clausius-Clapeyron)
- Enhanced condensate loading in warmer environments
- Faster microphysical process rates at higher water contents

These competing effects largely cancel, leaving $\tau^{-1}_{\text{cloud}}$ approximately **unchanged** on isotherms.

---

## 3. The Complete Budget

We can now construct a complete accounting of the cloud fraction response:

| Component | Response on Isotherms | Physical Mechanism |
|-----------|----------------------|-------------------|
| Mass flux $\omega$ | -5%/K | Stability increase ($S \uparrow$) while $Q$ constant |
| Pressure thickness $\Delta p$ | -3%/K | Lower density at constant $T$ |
| $\tau^{-1}_{\text{sub}} = \omega/\Delta p$ | -2%/K | Partial compensation |
| $\tau^{-1}_{\text{cloud}}$ | ~0%/K | Sinks and $\Delta q$ both increase |
| Cloud Fraction CF | **-1%/K** | Net residual after compensations |

Alternatively, in the $\text{CF} = \delta \cdot M \cdot \tau$ framework:

| Component | Response on Isotherms |
|-----------|----------------------|
| Mass flux $M$ | -5%/K |
| Fractional detrainment $\delta$ | +4%/K |
| Cloud lifetime $\tau$ | ~0%/K |
| Cloud Fraction CF | **-1%/K** |

The fractional detrainment increases because $\delta \sim 1/\Delta p$, and $\Delta p$ decreases as clouds exist at lower pressures (higher altitudes) at the same temperature in a warmer climate.

---

## 4. Physical Interpretation

### 4.1 Why the Mass Flux Decrease is Buffered

The fundamental insight is that cloud fraction depends not on mass flux alone, but on the **divergence of mass flux per unit pressure thickness**. Consider a convective tower detraining at the anvil level:

1. In a warmer climate, the same isotherm (e.g., 220 K) exists at lower pressure
2. The mass flux at that isotherm is reduced due to enhanced stability
3. However, the pressure thickness over which detrainment occurs is also reduced
4. The detrained mass per unit volume (not per unit pressure) is less reduced than the mass flux alone would suggest

This is analogous to squeezing the same amount of toothpaste through a thinner tube - the flux decreases, but the concentration can remain similar.

### 4.2 The Role of Cloud Microphysics

Cloud lifetime ($\tau_{\text{cloud}}$) exhibits remarkable stability on isotherms because:

1. **Clausius-Clapeyron scaling**: At constant temperature, saturation vapor pressure (and thus absolute humidity) is fixed. This constrains the thermodynamic environment for cloud processes.

2. **Compensating microphysical responses**: While warmer climates have more condensate (from enhanced moisture flux), the microphysical sinks (sedimentation, precipitation formation) also scale with condensate amount.

3. **Temperature-dependent processes**: Many microphysical processes (e.g., ice crystal growth, aggregation) depend primarily on temperature rather than pressure, making isothermal analysis particularly clean.

### 4.3 Altitude Dependence

The analysis reveals important altitude-dependent behavior:

- **Upper troposphere (T < 240 K)**: The framework holds cleanly. Both $\delta \cdot \omega$ and $\tau$ show minimal change, explaining the weak CF response.

- **Middle troposphere (250-273 K)**: More complex behavior emerges. Cloud lifetime shows significant SST-dependent variations that may reflect:
  - Mixed-phase microphysics complications
  - Interaction between liquid and ice processes
  - Possibly non-negligible updraft area fraction ($\sigma$)

The full balance equation including updraft contribution is:

$$\sigma \cdot \frac{M_c}{\Delta p} + \text{CF} \cdot \tau^{-1}_{\text{cloud}} = (1 - \text{CF}) \cdot \tau^{-1}_{\text{sub}}$$

where $\sigma$ is the updraft area fraction and $M_c/\Delta p$ can be thought of as $\tau^{-1}_{\text{inject}}$, the injection timescale.

---

## 5. Microphysical Budget Details

For completeness, we document the relevant microphysical terms in the P3 microphysics scheme (Morrison & Milbrandt, 2015):

**Cloud Sources:**
- Condensation (vapor $\rightarrow$ liquid/ice)
- Advection (horizontal/vertical transport of cloud into the layer)
- Rain freezing (rain $\rightarrow$ ice, counted as cloud source since ice is "cloud")

**Cloud Sinks:**
- Autoconversion ($q_c \rightarrow q_r$)
- Accretion ($q_c + q_r \rightarrow q_r$)
- Sedimentation (gravitational settling of ice)
- Sublimation/evaporation (ice/liquid $\rightarrow$ vapor)
- Melting (ice $\rightarrow$ rain)

From the P3 tendency equations:

$$\frac{\partial q_i}{\partial t}\bigg|_{P3} = \text{vap\_dep} - \text{sublim} + \text{berg} + \text{cloud\_freeze} + \text{rain\_freeze} - \text{melt} - \text{sed}$$

$$\frac{\partial q_c}{\partial t}\bigg|_{P3} = -\text{qc2qr} - \text{cloud\_freeze} - \text{berg}$$

These can be combined to constrain unmeasured terms when full process-level output is unavailable.

---

## 6. Discussion

### 6.1 Robustness Across Models

The mass flux constraint ($\omega = Q/S$) is robust across climate models (Held & Soden, 2006). The present GSRM results affirm this fundamental relationship. The novelty here lies in:

1. Demonstrating the constraint in a GSRM with explicit convection
2. Explaining why cloud fraction does not track mass flux one-to-one
3. Providing a simple diagnostic framework connecting thermodynamics to cloud fraction

### 6.2 Implications for Cloud Feedbacks

If cloud fraction decreases only weakly on isotherms (~1%/K vs. ~5%/K), several implications follow:

1. **The FAT hypothesis extended**: The Fixed Anvil Temperature (FAT) hypothesis (Hartmann & Larson, 2002) explains why anvil cloud temperature remains approximately constant with warming. Our analysis adds that anvil cloud *fraction* also remains approximately constant on isotherms.

2. **Reduced negative feedback**: A weaker CF decrease on isotherms implies a weaker negative shortwave cloud feedback than mass flux arguments alone would suggest.

3. **Importance of pressure coordinates**: Analyses performed in pressure coordinates may miss the compensating $\Delta p$ effects. Isothermal analysis provides cleaner physical interpretation.

### 6.3 Caveats and Future Work

Several aspects require further investigation:

1. **Middle troposphere complexity**: The lifetime spike between 250-273 K appears sensitive to SST in ways that may not be physical, potentially reflecting:
   - Limitations of the steady-state assumption
   - Non-negligible rain advection (not included in current budget)
   - Mixed-phase microphysics complexities

2. **Model dependence**: The partial compensation between sinks and $\Delta q$ in $\tau^{-1}_{\text{cloud}}$ may be model-specific. Multi-model analysis is needed.

3. **Observational validation**: The framework makes testable predictions about how cloud properties should vary with temperature that could be evaluated against satellite observations.

---

## 7. Summary

We have presented a thermodynamically grounded framework for understanding cloud fraction response to warming. The key findings are:

1. **Cloud fraction is the ratio of timescales**: $\text{CF} \approx \tau_{\text{cloud}} / \tau_{\text{sub}}$

2. **The mass flux decrease is buffered by density decrease**: On isotherms, clouds exist at lower pressure in a warmer climate, reducing $\Delta p$ and partially compensating the reduced mass flux.

3. **Cloud lifetime is approximately conserved on isotherms**: Both cloud sources and sinks scale with moisture content, leaving their ratio unchanged.

4. **Quantitative budget**:
   - Mass flux: -5%/K
   - $\Delta p$ compensation: +3%/K
   - Microphysical compensation: +1%/K
   - Net CF change: **-1%/K**

The framework provides a simple but powerful lens for interpreting cloud fraction changes in warming climates, unifying thermodynamic constraints with microphysical processes.

---

## References

Beydoun, H., Caldwell, P. M., Hannah, W. M., & Donahue, A. S. (2021). Dissecting anvil cloud response to sea surface warming. *Geophysical Research Letters*, 48(15), e2021GL094049.

Bony, S., Stevens, B., Frierson, D. M. W., Jakob, C., Kageyama, M., Pincus, R., ... & Webb, M. J. (2015). Clouds, circulation and climate sensitivity. *Nature Geoscience*, 8(4), 261-268.

Bony, S., Bellon, G., Klocke, D., Sherwood, S., Fermepin, S., & Denvil, S. (2013). Robust direct effect of carbon dioxide on tropical circulation and regional precipitation. *Nature Geoscience*, 6(6), 447-451.

Hartmann, D. L., & Larson, K. (2002). An important constraint on tropical cloud-climate feedback. *Geophysical Research Letters*, 29(20), 1951.

Held, I. M., & Soden, B. J. (2006). Robust responses of the hydrological cycle to global warming. *Journal of Climate*, 19(21), 5686-5699.

Kuang, Z., & Hartmann, D. L. (2007). Testing the fixed anvil temperature hypothesis in a cloud-resolving model. *Journal of Climate*, 20(10), 2051-2057.

Morrison, H., & Milbrandt, J. A. (2015). Parameterization of cloud microphysics based on the prediction of bulk ice particle properties. Part I: Scheme description and idealized tests. *Journal of the Atmospheric Sciences*, 72(1), 287-311.

Sherwood, S. C., Webb, M. J., Annan, J. D., Armour, K. C., Forster, P. M., Hargreaves, J. C., ... & Zelinka, M. D. (2020). An assessment of Earth's climate sensitivity using multiple lines of evidence. *Reviews of Geophysics*, 58(4), e2019RG000678.

Vecchi, G. A., & Soden, B. J. (2007). Global warming and the weakening of the tropical circulation. *Journal of Climate*, 20(17), 4316-4340.

Zelinka, M. D., Myers, T. A., McCoy, D. T., Po-Chedley, S., Caldwell, P. M., Ceppi, P., ... & Taylor, K. E. (2020). Causes of higher climate sensitivity in CMIP6 models. *Geophysical Research Letters*, 47(1), e2019GL085782.

---

## Appendix A: Symbol Definitions

| Symbol | Definition | Units |
|--------|------------|-------|
| CF | Cloud fraction | dimensionless |
| $\omega$ | Vertical velocity in pressure coordinates | Pa s$^{-1}$ |
| $Q$ | Radiative heating rate | K s$^{-1}$ |
| $S$ | Static stability ($-\partial T / \partial p$) | K Pa$^{-1}$ |
| $M$ | Convective mass flux | kg m$^{-2}$ s$^{-1}$ |
| $\delta$ | Fractional detrainment rate | Pa$^{-1}$ |
| $\tau$ | Cloud lifetime | s |
| $\tau_{\text{sub}}$ | Subsidence timescale | s |
| $\tau_{\text{cloud}}$ | Cloud dissipation timescale | s |
| $\Delta p$ | Pressure thickness of cloud layer | Pa |
| $\Delta q$ | Detrained condensate mixing ratio | kg kg$^{-1}$ |
| $q_i$, $q_c$, $q_r$ | Ice, cloud, rain mixing ratios | kg kg$^{-1}$ |
| $\sigma$ | Updraft area fraction | dimensionless |

---

## Appendix B: Key Equations Summary

**Thermodynamic constraint on mass flux:**
$$\omega = \frac{Q}{S}$$

**Cloud fraction from timescale ratio:**
$$\text{CF} = \frac{\tau_{\text{cloud}}}{\tau_{\text{sub}}} = \frac{\tau^{-1}_{\text{sub}}}{\tau^{-1}_{\text{cloud}}}$$

**Inverse subsidence timescale:**
$$\tau^{-1}_{\text{sub}} = \frac{\omega_{\text{sub}}}{\Delta p} = \frac{Q/S}{\Delta p}$$

**Inverse cloud timescale:**
$$\tau^{-1}_{\text{cloud}} = \frac{\text{Sinks}}{\Delta q}$$

**Full mass balance (including updrafts):**
$$\sigma \cdot \frac{M_c}{\Delta p} + \text{CF} \cdot \tau^{-1}_{\text{cloud}} = (1 - \text{CF}) \cdot \tau^{-1}_{\text{sub}}$$

---

*Document generated from Slack discussion thread, March 2026.*
*Slack thread: https://pyaos.slack.com/archives/C0AMDP1F0JU/p1773449065038849*
