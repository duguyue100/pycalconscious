# Author: Hu Yuhuang
# Date  : 2014-09-18

import numpy as np;
from scipy.linalg import block_diag

def calCovariance(X):
    """
    this function calculates covariance matrix of X
    
    :type X: 1-d or 2-d array
    :param X: assume to be a multivariate Gaussian distribution
              variables are row vectors
    """
    return np.cov(X);

def calCrossCovariance(X, Y):
    """
    This function calculates cross covariance between X and Y.
    
    :type X: 1-d or 2-d array
    :param X: assume to be a multivariate Gaussian distribution
              variables are row vectors
              
    :type Y: 1-d or 2-d array
    :param Y: assume to be a multivariate Gaussian distribution
              variables are row vectorss
              
    X and Y share same dimensions.
    """
    N=X.shape[0];
    
    cc=np.zeros((N, N));
    
    for i in range(N):
        for j in range(N):
            cc[i,j]=np.cov(X[i,:], Y[j,:])[0,1];
    
    return cc;
    
def calConditionalCovariance(X,Y):
    """
    Calculate Sigma(X|Y)
    
    :type X: 1-d or 2-d array
    :param X: assume to be a multivariate Gaussian distribution
              variables are row vectors
              
    :type Y: 1-d or 2-d array
    :param Y: assume to be a multivariate Gaussian distribution
              variables are row vectorss
              
    X and Y share same dimensions.
    """
    
    return calCovariance(X)-calCrossCovariance(X,Y).dot(np.linalg.inv(calCovariance(Y))).dot(calCrossCovariance(X,Y).T);

def calInformation(X):
    """
    Calculate H(X)
    
    :type X: 1-d or 2-d array
    :param X: assume to be a multivariate Gaussian distribution
              variables are row vectors
    """
    
    return (1.0/2.0)*np.log(np.linalg.det(calCovariance(X)))+(1.0/2.0)*X.shape[0]*np.log(2*np.pi*np.e); 

def calConditionalInformation(X, Y):
    """
    Calculate H(X|Y)
    
    :type X: 1-d or 2-d array
    :param X: assume to be a multivariate Gaussian distribution
              variables are row vectors
              
    :type Y: 1-d or 2-d array
    :param Y: assume to be a multivariate Gaussian distribution
              variables are row vectorss
              
    X and Y share same dimensions.
    """
    
    return (1.0/2.0)*np.log(np.linalg.det(calConditionalCovariance(X,Y)))+(1.0/2.0)*X.shape[0]*np.log(2*np.pi*np.e);
    

def calMutualInformation(X,Y):
    """
    Calculate I(X;Y)
    
    :type X: 1-d or 2-d array
    :param X: assume to be a multivariate Gaussian distribution
              variables are row vectors
              
    :type Y: 1-d or 2-d array
    :param Y: assume to be a multivariate Gaussian distribution
              variables are row vectorss
              
    X and Y share same dimensions.
    """
    
    return (1.0/2.0)*np.log(np.linalg.det(calCovariance(X))/np.linalg.det(calConditionalCovariance(X,Y)));

def calDiagCovariance(M):
    """
    Calculate diagonal covariance of subsystems in M
    
    :type M: a 3-dimension array
    :param M: represents subsystems
    """
    
    cM=np.zeros((M.shape[0],M.shape[0],M.shape[2]));
    
    for i in range(M.shape[2]):
        cM[:,:,i]=calCovariance(M[:,:,i]);
        
    dM=cM[:,:,0];
    for i in range(1,cM.shape[2]):
        dM=block_diag(dM, cM[:,:,i]);
    
    return dM;

def calDiagCrossCovariance(M1, M2):
    """
    Calculate diagonal cross covariance of subsystems M1, M2
    (usually M^t and M^(t-tau))
    
    :type M1: a 3-d array
    :param M1: represents subsystem
    
    :type M1: a 3-d array
    :param M2: represents another subsystem
    """
    cM=np.zeros((M1.shape[0],M1.shape[0],M1.shape[2]));
    
    for i in range(M1.shape[2]):
        cM[:,:,i]=calCrossCovariance(M1[:,:,i], M2[:,:,i]);
    
    dM=cM[:,:,0];
    for i in range(1, cM.shape[2]):
        dM=block_diag(dM, cM[:,:,i]);
    
    return dM;

def calDiagConditionalCovariance(M1, M2):
    """
    Calculate diagonal conditional covariance of subsystems M1, M2
    (usually M^t and M^(t-tau))
    
    :type M1: a 3-d array
    :param M1: represents subsystem
    
    :type M1: a 3-d array
    :param M2: represents another subsystem
    """
    cM=np.zeros((M1.shape[0],M1.shape[0],M1.shape[2]));
    
    for i in range(M1.shape[2]):
        cM[:,:,i]=calConditionalCovariance(M1[:,:,i], M2[:,:,i]);
    
    dM=cM[:,:,0];
    for i in range(1, cM.shape[2]):
        dM=block_diag(dM, cM[:,:,i]);
    
    return dM;

def calQ(Xt, Xt_tau, Mt, Mt_tau, beta):
    """
    Calculate Q indicated in paper.
    
    :type Xt: 2-d array
    :param Xt: a state on time t: X^t
    
    :type Xt_tau: 2-d array
    :param Xt_tau: a state on time t-tau: X^(t-tau)
    
    :type Mt: 3-d array
    :param Mt: subsystems on time t: M^t
    
    :type Mt_tau: 3_array
    :param Mt_tau: subsystems on time t-tau: M^(t-tau)
    
    :type beta: constant
    :param beta: constant which maximize accuracy of the mismatched decoding.

    """
    
    diagCondXtXttau=np.linalg.inv(calDiagConditionalCovariance(Mt, Mt_tau));
    diagCrossXtXttau=calDiagCrossCovariance(Mt, Mt_tau);
    diagXttau=np.linalg.inv(calDiagCovariance(Mt_tau));
    
    return diagXttau+beta*(diagXttau.dot(diagCrossXtXttau.T).dot(diagCondXtXttau).dot(diagCrossXtXttau).dot(diagXttau));

def calR(Xt, Xt_tau, Mt, Mt_tau, beta):
    """
    Calculate Q indicated in paper.
    
    :type Xt: 2-d array
    :param Xt: a state on time t: X^t
    
    :type Xt_tau: 2-d array
    :param Xt_tau: a state on time t-tau: X^(t-tau)
    
    :type Mt: 3-d array
    :param Mt: subsystems on time t: M^t
    
    :type Mt_tau: 3_array
    :param Mt_tau: subsystems on time t-tau: M^(t-tau)
    
    :type beta: constant
    :param beta: constant which maximize accuracy of the mismatched decoding.

    """
    Q=calQ(Xt, Xt_tau, Mt, Mt_tau, beta);
    diagCondXtXttau=np.linalg.inv(calDiagConditionalCovariance(Mt, Mt_tau));
    diagCrossXtXttau=calDiagCrossCovariance(Mt, Mt_tau);
    diagXttau=np.linalg.inv(calDiagCovariance(Mt_tau));
    
    return beta*diagCondXtXttau-beta**2*(diagCondXtXttau.T.dot(diagCrossXtXttau).dot(diagXttau).dot(Q).dot(diagXttau).dot(diagCrossXtXttau.T).dot(diagCondXtXttau));

def calIstar(Xt, Xt_tau, Mt, Mt_tau, beta):
    """
    Calculate I^*(X^(t-tau); X^t) indicated in paper.
    
    :type Xt: 2-d array
    :param Xt: a state on time t: X^t
    
    :type Xt_tau: 2-d array
    :param Xt_tau: a state on time t-tau: X^(t-tau)
    
    :type Mt: 3-d array
    :param Mt: subsystems on time t: M^t
    
    :type Mt_tau: 3_array
    :param Mt_tau: subsystems on time t-tau: M^(t-tau)
    
    :type beta: constant
    :param beta: constant which maximize accuracy of the mismatched decoding.

    """
    Q=calQ(Xt, Xt_tau, Mt, Mt_tau, beta);
    R=calR(Xt, Xt_tau, Mt, Mt_tau, beta);
    Istar_p1=(1.0/2.0)*np.trace(calCovariance(Xt_tau).dot(R));
    Istar_p2=(1.0/2.0)*np.log(np.linalg.det(Q)/(np.linalg.det(calCovariance(Xt_tau))*np.linalg.det(calDiagConditionalCovariance(Mt, Mt_tau))**beta));
    Istar_p3=(beta*Xt.shape[0])/2.0*np.log(2*np.pi);
    Istar_p4=0;
    for i in range(Mt.shape[2]):
        Istar_p4=Istar_p4+calConditionalInformation(Mt[:,:,i], Mt_tau[:,:,i]);
    Istar_p4=beta*Istar_p4;
    
    return Istar_p1-Istar_p2+Istar_p3-Istar_p4;

def calI(X_t, Xt_tau):
    """
    Calculate I(X^(t-tau); X_t) indicated in paper.
    
    :type Xt: 2-d array
    :param Xt: a state on time t: X^t
    
    :type Xt_tau: 2-d array
    :param Xt_tau: a state on time t-tau: X^(t-tau)
    """
    
    return calMutualInformation(Xt_tau, X_t);

def calIntegratedInformation(Xt, Xt_tau, Mt, Mt_tau, beta):
    """
    Calculate phi^*=I-I^* indicated in paper.
    
    :type Xt: 2-d array
    :param Xt: a state on time t: X^t
    
    :type Xt_tau: 2-d array
    :param Xt_tau: a state on time t-tau: X^(t-tau)
    
    :type Mt: 3-d array
    :param Mt: subsystems on time t: M^t
    
    :type Mt_tau: 3_array
    :param Mt_tau: subsystems on time t-tau: M^(t-tau)
    
    :type beta: constant
    :param beta: constant which maximize accuracy of the mismatched decoding.

    """
    
    return calI(Xt, Xt_tau)-calIstar(Xt, Xt_tau, Mt, Mt_tau, beta);

## help function for gradient descent

def calDIstarDbeta(Xt, Xt_tau, Mt, Mt_tau, beta):
    """
    Calculate (d I^*(beta)/d beta) indicated in paper.
    
    :type Xt: 2-d array
    :param Xt: a state on time t: X^t
    
    :type Xt_tau: 2-d array
    :param Xt_tau: a state on time t-tau: X^(t-tau)
    
    :type Mt: 3-d array
    :param Mt: subsystems on time t: M^t
    
    :type Mt_tau: 3_array
    :param Mt_tau: subsystems on time t-tau: M^(t-tau)
    
    :type beta: constant
    :param beta: constant which maximize accuracy of the mismatched decoding.
    """
    
    Q=calQ(Xt, Xt_tau, Mt, Mt_tau, beta);
    dQdet=calDQdetDbeta(Xt, Xt_tau, Mt, Mt_tau, beta, Q);
    dR=calDRDbeta(Xt, Xt_tau, Mt, Mt_tau, beta, Q);
    
    dI_part1=(1.0/2.0)*np.trace(calCovariance(Xt_tau).dot(dR));
    dI_part2=(1.0/(2.0*np.linalg.det(Q)))*dQdet;
    dI_part3=(1.0/2.0)*np.log(np.linalg.det(calDiagConditionalCovariance(Mt, Mt_tau)));
    dI_part4=Xt.shape[0]/2.0*np.log(2*np.pi);
    
    dI_part5=0;
    for i in range(Mt.shape[2]):
        dI_part5=dI_part5+calConditionalInformation(Mt[:,:,i], Mt_tau[:,:,i]);
    
    return dI_part1-dI_part2+dI_part3+dI_part4-dI_part5;

def calDRDbeta(Xt, Xt_tau, Mt, Mt_tau, beta, Q):
    """
    Calculate (d I^*(beta)/d beta) indicated in paper.
    
    :type Xt: 2-d array
    :param Xt: a state on time t: X^t
    
    :type Xt_tau: 2-d array
    :param Xt_tau: a state on time t-tau: X^(t-tau)
    
    :type Mt: 3-d array
    :param Mt: subsystems on time t: M^t
    
    :type Mt_tau: 3_array
    :param Mt_tau: subsystems on time t-tau: M^(t-tau)
    
    :type beta: constant
    :param beta: constant which maximize accuracy of the mismatched decoding.
    
    :type Q: matrix
    :param Q: from calQ(Xt, Xt_tau, Mt, Mt_tau, beta)
    """
    diagCondXtXttau=calDiagConditionalCovariance(Mt, Mt_tau);
    diagCondXtXttauInv=np.linalg.inv(diagCondXtXttau);
    
    diagCrossXtXttau=calDiagCrossCovariance(Mt, Mt_tau);
    diagXttau=np.linalg.inv(calDiagCovariance(Mt_tau));
    
    dR_part1=diagCondXtXttau;
    dR_part2=2*beta*diagCondXtXttauInv.T.dot(diagCrossXtXttau).dot(diagXttau).dot(Q).dot(diagXttau).dot(diagCrossXtXttau.T).dot(diagCondXtXttauInv);
    dR_part3=beta**2*diagCondXtXttauInv.T.dot(diagCrossXtXttau).dot(diagXttau).dot(calDQDbeta(Xt, Xt_tau, Mt, Mt_tau, beta, Q)).dot(diagXttau).dot(diagCrossXtXttau.T).dot(diagCondXtXttauInv);
    
    return dR_part1-dR_part2-dR_part3;

def calDQdetDbeta(Xt, Xt_tau, Mt, Mt_tau, beta, Q):
    """
    Calculate (d |Q|/d beta) indicated in paper.
    
    :type Xt: 2-d array
    :param Xt: a state on time t: X^t
    
    :type Xt_tau: 2-d array
    :param Xt_tau: a state on time t-tau: X^(t-tau)
    
    :type Mt: 3-d array
    :param Mt: subsystems on time t: M^t
    
    :type Mt_tau: 3_array
    :param Mt_tau: subsystems on time t-tau: M^(t-tau)
    
    :type beta: constant
    :param beta: constant which maxcalDiagConditionalCovariance(Mt, Mt_tau)imize accuracy of the mismatched decoding.
    
    :type Q: matrix
    :param Q: from calQ(Xt, Xt_tau, Mt, Mt_tau, beta)
    """
    return np.linalg.det(Q)*np.trace(np.linalg.inv(Q)*calDQDbeta(Xt, Xt_tau, Mt, Mt_tau, beta, Q));

def calDQDbeta(Xt, Xt_tau, Mt, Mt_tau, beta, Q):
    """
    Calculate (d Q/d beta) indicated in paper.
    
    :type Xt: 2-d array
    :param Xt: a state on time t: X^t
    
    :type Xt_tau: 2-d array
    :param Xt_tau: a state on time t-tau: X^(t-tau)
    
    :type Mt: 3-d array
    :param Mt: subsystems on time t: M^t
    
    :type Mt_tau: 3_array
    :param Mt_tau: subsystems on time t-tau: M^(t-tau)
    
    :type beta: constant
    :param beta: constant which maximize accuracy of the mismatched decoding.
    
    :type Q: matrix
    :param Q: from calQ(Xt, Xt_tau, Mt, Mt_tau, beta)
    """
    
    diagXttau=np.linalg.inv(calDiagCovariance(Mt_tau));
    diagCrossXtXttau=calDiagCrossCovariance(Mt, Mt_tau);
    
    return -1*Q.dot(diagXttau).dot(diagCrossXtXttau.T).dot(np.linalg.inv(calDiagConditionalCovariance(Mt, Mt_tau))).dot(diagCrossXtXttau).dot(diagXttau).dot(Q);