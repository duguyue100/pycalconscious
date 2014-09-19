PyCalConscious
==============

A Python implementation of computing integrated information based on mismatched decoding.

The implemented method can be found at:

_Measuring the level of consciousness based on the amount of integrated information_

by: Masafumi Oizumi, Toru Yanagawa, Shun-ichi Amari, Naotsugu Tsuchiya, Naotaka Fujii

This repo is developed under:
+ Ubuntu 12.04 64bit
+ Python 2.7.3
+ `numpy` 1.6.1
+ `scipy` 0.9.0

__This implementation is carried out based on my own understanding of the method, please point out my errors if there is any__

##Updates

+ The algorithm of mismatched decoding is updated [2014-09-18] [2014-09-19 TESTED]
+ Some tests are updated [2014-09-18] [2014-09-19 TESTED]

##Notes

+ All functions are contained in `calconscious_lib.py`, you can simply import to your script.

+ This code is to measure integrated information between state `X^t` and `X^(t-tau)` based on mismatched decoding.

+ A state `X` is a N*M matrix where N is number of bipolar re-referenced electrodes (or similar idea) and M is dimension of bipolor re-referenced signals (in paper, M=64).

+ You also need to prepare data for subsystems `M^t` and `M^(t-tau)` (The test is simply using data generated from Gaussian distribution and divides them into some parts, it's not so clear to me how this division is performed).

+ The method also involves a Gradient Decent process to find a constant `beta` which maximizes accuracy of mismatched decoding.

+ When the data is ready and `beta` is optimized, then you can simply call:
   ```
   calIntegratedInformation(Xt, Xt_tau, Mt, Mt_tau, beta)
   ```
   This function will return a real value which indicates the integrated information.

##Contacts

Hu Yuhuang

_No. 42, North, Flatland_

Email: duguyue100@gmail.com
