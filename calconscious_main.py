import numpy as np;
import calconscious_lib as ccl;


"""
This test, since I don't have any real data, so I simply use numpy's
random function to generate Gaussian-like signals to test the correction of implementation.

Xt here is a 8*64 array which consists of 8 samples (X^t)
Xt_tau is a 8*64 array which consists of 8 samples (X^(t-tau))

I simply divide them in two parts, and
Mt here is a 4*64*2 array where each part has 4 samples (M^t)
Mt_tau is a 4*64*2 array where each part has 4 samples (M^(t-tau))

beta is a constant

I didn't perform Gradient descent in this test, but I tested all the functions,
they are all working right now.

This implementation is carried out based on my understanding,
please point it out if you find any problems.
"""

mu, sigma = 0, 0.1;
Xt = np.array([np.random.normal(mu, sigma, 64),
               np.random.normal(mu, sigma, 64),
               np.random.normal(mu, sigma, 64),
               np.random.normal(mu, sigma, 64),
               np.random.normal(mu, sigma, 64),
               np.random.normal(mu, sigma, 64),
               np.random.normal(mu, sigma, 64),
               np.random.normal(mu, sigma, 64)]);
            
Xt_tau = np.array([np.random.normal(mu, sigma, 64),
                   np.random.normal(mu, sigma, 64),
                   np.random.normal(mu, sigma, 64),
                   np.random.normal(mu, sigma, 64),
                   np.random.normal(mu, sigma, 64),
                   np.random.normal(mu, sigma, 64),
                   np.random.normal(mu, sigma, 64),
                   np.random.normal(mu, sigma, 64),]);
              

Mt=np.zeros((4,64,2));

Mt[:,:,0]=np.array([np.random.normal(mu, sigma, 64),
                    np.random.normal(mu, sigma, 64),
                    np.random.normal(mu, sigma, 64),
                    np.random.normal(mu, sigma, 64)]);
Mt[:,:,1]=np.array([np.random.normal(mu, sigma, 64),
                    np.random.normal(mu, sigma, 64),
                    np.random.normal(mu, sigma, 64),
                    np.random.normal(mu, sigma, 64)]);

Mt_tau=np.zeros((4,64,2));
            
Mt_tau[:,:,0]=np.array([np.random.normal(mu, sigma, 64),
                        np.random.normal(mu, sigma, 64),
                        np.random.normal(mu, sigma, 64),
                        np.random.normal(mu, sigma, 64)]);
Mt_tau[:,:,1]=np.array([np.random.normal(mu, sigma, 64),
                        np.random.normal(mu, sigma, 64),
                        np.random.normal(mu, sigma, 64),
                        np.random.normal(mu, sigma, 64)]);
     

beta=0.2;

print ccl.calIntegratedInformation(Xt, Xt_tau, Mt, Mt_tau, beta);

print ccl.calDIstarDbeta(Xt, Xt_tau, Mt, Mt_tau, beta);