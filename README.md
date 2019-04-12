# Final_Project
IS590PR Spring 2019.

## Project Title
**Bus Transit Simulation with Focus On Clumping Reduction**

## Team
Dhruman Shah github: *dhruman-shah*
Pranay Parmar github: *pranayparmar*
Sarvani Putta github: *sarvaniputta*

## Background
Bus transit systems often face the problem of clumping where buses that are scheduled to be equally spaced in time tend to arrive at a stop at the same time.
This is often due to the snowballing effect of delays for a bus which causes it to pick more passengers leading to more delays. The bus behind on the contrary needs to pick up
fewer passengers and typically tends to speed up. This project considers a few hypothesis for reduction of this clumping phenomena and verifies them with simulation. A baseline
simulation with no clumping is designed and delays added to obtain clumping. With this clumped system as our baseline, we simulate our hypothesis strategies and observe if clumping
is reduced.

## Hypothesis
1) *Stopping for a minimum time at every stop irrespective of passenger presence reduces clumping.* The idea is buses behind delayed buses do not rush ahead due to lack of onboarding passengers

2) *Overtaking a delayed bus reduces clumping.* The idea is by letting the faster bus pick up passengers at the next stop and reducing the onboarding delay on the next bus.

