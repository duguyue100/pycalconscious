import numpy as np;
from scipy.linalg import block_diag;
import calconscious_lib as ccl;


### input
mu, sigma = 0, 0.1;
x = np.array([np.random.normal(mu, sigma, 64),
              np.random.normal(mu, sigma, 64)]);
            
y = np.array([np.random.normal(mu, sigma, 64),
              np.random.normal(mu, sigma, 64)]);
            
print ccl.calCovariance(x).T;

print ccl.calConditionalCovariance(x, y);

print np.linalg.det(ccl.calCovariance(x))/np.linalg.det(ccl.calConditionalCovariance(x, y));

z=np.ones((2,2,3));
print z[:,:,:];

result=z[:,:,0];

for i in range(1,z.shape[2]):
    result=block_diag(result, z[:,:,i]);

print result;
 