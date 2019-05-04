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

**Hypothesis 2**: *Stopping for a minimum time at every stop irrespective of passenger presence reduces bunching.* The idea is buses behind delayed buses do not rush ahead due to lack of onboarding passengers.

**Hypothesis 3**: *Combining the hypothesis 1 & hypothesis 2 to see the effect on bunching*. The idea behing this is to observe the effect on bunching if we combine the above two hypothesis in order to see which hypothesis is better in reducing the bunching.

## Results
**Hypothesis 1:**
After simulating the first scenario, we observed that the bunching reduced by almost 65% against baseline & the average length of the queue of passengers reduced by 53% against baseline. 

**Hypothesis 2:**
After simulating the second scenario, we observed that allowing the bus to stop for a fixed time at certain stops did not have much effect on reducing the bunching. The percentage reduction in bunching was observed to be around 13% against the baseline whereas the reduction in average length of the queue was observed to be around 33% against the baseline.

**Hypothesis 3:**
After simulating the third scenario, we observed it's behaviour to be almost similar to that of the first scenario i.e. overtaking a delayed bus. The percentage reduction in bunching was s;ightly higher as compared to scenario 1, around 68% against the baseline. The percentage reduction in average queue length was observed to be 53% against the baseline. 




The results have been tabulated below:



+------------------+----------+--------------+--------------+--------------+
|    Parameters    | Baseline | Hypothesis 1 | Hypothesis 2 | Hypothesis 3 |
+------------------+----------+--------------+--------------+--------------+
|   Mean bunches   |  16.37   |     5.72     |    14.22     |     5.15     |
| Avg Queue Length |  12.22   |     5.69     |     8.18     |     5.7      |
+------------------+----------+--------------+--------------+--------------+




The percentage changes in bunching reduction and average queue length are tabulated below:



+------------------+--------------------------+--------------------------+--------------------------+
|    Parameters    | % Reduction Hypothesis 1 | % Reduction Hypothesis 2 | % Reduction Hypothesis 3 |
+------------------+--------------------------+--------------------------+--------------------------+
|   Mean bunches   |          65.06           |          13.13           |          68.54           |
| Avg Queue Length |          53.44           |          33.06           |          53.36           |
+------------------+--------------------------+--------------------------+--------------------------+

## Conclusion
Thus, from our analysis we can conclude by saying that overtaking a bus gives better results (65%) as compared to stopping for a fixed time at certain stops (13%). However, further analysis indicated that when these two hypotheses are combined we get a much better result (68%) in the reduction in number of bunches.


## Git URL
https://github.com/sarvaniputta/Final_Project

## References
https://en.wikipedia.org/wiki/Bus_bunching

