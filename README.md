# Final_Project
IS590PR Spring 2019.

## Project Title
**Bus Transit Simulation with Focus on Bunching Reduction**

## Team
- Dhruman Shah github: *dhruman-shah*
- Pranay Parmar github: *pranayparmar*
- Sarvani Putta github: *sarvaniputta*

## Background
Bus transit systems often face the problem of bunching where buses that are scheduled to be equally spaced in time tend to arrive at a stop at the same time.
This is often due to the snowballing effect of delays for a bus which causes it to pick more passengers leading to more delays. The bus behind on the contrary needs to pick up
fewer passengers and typically tends to speed up. This project considers a few hypothesis for reduction of this bunching phenomena and verifies them with simulation. A baseline
simulation with no bunching is designed and delays added to obtain bunching. With this bunched system as our baseline, we simulate our hypothesis strategies and observe if bunching
is reduced.

## Hypothesis
**Hypothesis 1**: *Overtaking a delayed bus reduces bunching.* The idea is by letting the faster bus pick up passengers at the next stop and reducing the onboarding delay on the next bus.

**Hypothesis 2**: *Stopping a faster moving bus for a minimum time at certain stops irrespective of passenger presence reduces bunching.* The idea is buses behind delayed buses do not rush ahead due to lack of onboarding passengers.

**Hypothesis 3**: *Combining the hypothesis 1 & hypothesis 2 to see the effect on bunching*. Combining hypothesis 1 & 2 will produce the best results.

## Results
**Hypothesis 1:**
After simulating the first scenario, we observed that the bunching reduced by almost 65% against baseline & the average length of the passengers queue reduced by 53% against baseline. 

**Hypothesis 2:**
After simulating the second scenario, we observed that the percentage reduction in bunching was observed to be around 13% against the baseline whereas the reduction in average length of the passenger queue was around 33% against the baseline. 

**Hypothesis 3:**
After simulating the third scenario, we observed the percentage reduction in bunching was slightly better as compared to scenario 1, around 68% against the baseline. The percentage reduction in average queue length was observed to be 53% against the baseline. 

The aggregate summary of the simulations have been tabulated below:

   | Parameters | Baseline | Hypothesis 1 | Hypothesis 2 | Hypothesis 3 |
  | -------------- | ---------- | -------------- | --------------- | ------------- |
  | Mean bunches   |   16.37  |     5.72     |    14.22      |    5.15     |   
| Avg Queue Length |   12.22  |     5.69     |     8.18      |    5.7      |

The percentage changes in bunching reduction and average queue length are tabulated below:

| Parameters    | % Reduction Hypothesis 1 | % Reduction Hypothesis 2 | % Reduction Hypothesis 3 |
| ------------------ | -------------------------- | -------------------------- | -------------------------- |
|   Mean bunches   |          65.06           |          13.13           |          68.54           |
| Avg Queue Length |          53.44           |          33.06           |          53.36           |


## Conclusion
The results on simulating the model for 1000 times confirms that overtaking produces the better result as compared to the second reduction technique of stopping the bus for a fixed dwell time at certain stops. However, on combining both the techniques produces the best results with improvements in both, the reduction in bunches as well as passenger queue length.

## Git URL
https://github.com/sarvaniputta/Final_Project

## References
https://en.wikipedia.org/wiki/Bus_bunching

