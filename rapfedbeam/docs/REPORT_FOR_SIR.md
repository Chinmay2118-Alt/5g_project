# Draft message to Sir

Subject: RAP-FedWMMSE implemented — results and points for your review (Group 12)

Respected Sir,

We have implemented the RAP-FedWMMSE algorithm you sent, together with all baselines, and produced the 14
comparison graphs. The main results on the default network (8 APs, 2 relays, 8 UEs, 20 drops over 4 deployments)
are below.

| Method | PF utility | Mean rate (bit/s/Hz) | 5th-percentile rate | Time per drop |
|---|---|---|---|---|
| Centralized WMMSE-SCA/AO | 9.33 | 4.36 | 0.71 | 57 s |
| **RAP-FedWMMSE (proposed)** | **7.51** | **3.48** | **0.58** | **0.3 s** |
| Centralized WMMSE, no relay | 7.71 | 4.13 | 0.17 | 14 s |
| Unfolded WMMSE, untrained | 3.61 | 2.04 | 0.40 | 0.07 s |
| Local-WMMSE / No-FL | −3.28 | 0.87 | 0.24 | 1.5 s |

**Main findings**
1. RAP-FedWMMSE reaches about 80% of the centralized mean rate, using only local CSI and coordination
   statistics. Its inference is about 180 times faster.
2. Its 5th-percentile rate is 3.5 times that of the centralized method without relays.
3. On the default network, learning helps at every depth. With 4–10 unfolded layers, the trained model beats the
   same layers untrained by 0.9 to 1.5 in PF utility.
4. Under channel aging, the robust training of Step 10 shows. RAP-FedWMMSE scores higher than the centralized
   method on average at every ρ up to 0.95, although the confidence bands overlap.
5. Topology-aware personalized aggregation reached the highest validation utility, using about half of FedAvg's
   communication (11 MB against 20 MB). With 4 deployments, however, the differences between personalized,
   FedAvg and no federation are within statistical noise.
6. On a small network, the exact MINLP solver confirms that the centralized method is near-optimal (within 0.05).
   RAP-FedWMMSE is 0.53 below the certified bound on average. Here, unlike on the default network, the untrained
   layers did better (0.29 below the bound), largely because of one drop where the trained model was 4.7 below.

**Points where we had to interpret the algorithm, for your review**
1. In Step 9, r_k = min(C_D, C_AR) makes r_k − C_AR ≤ 0 always, so the DF dual price would never increase. We
   priced C_D − C_AR instead, which is positive exactly when the relay hop is the bottleneck.
2. For each node's gradient to be exact, ζ must carry the per-stream effective gains (about K² values per phase).
   At our size this is more signalling per block than raw CSI, although it does not grow with the antenna count.
3. We used λ_reg as a learned scaling of each node's power price. With neutral values, the layer reduces exactly
   to classical WMMSE.
4. In Step 1, a relay is eligible for a user only if its relay-to-user link beats the direct link. This rule is
   relaxed by the user's QoS price. Without it, the additive score relayed users whose relay link was weak.

We would be grateful for your feedback on these choices. We would also like your view on the network size and the
number of drops for the final report.

Sincerely,
Group 12
