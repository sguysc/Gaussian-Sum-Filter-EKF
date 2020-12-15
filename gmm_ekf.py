#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Implementation of the Gaussian Sum Filter for the problem
of a point robot driving through a corridor with faulty lidar sensor

22 Nov, 2020
"""
import numpy as np
import scipy.interpolate as interpolate
from scipy import stats
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from scipy.spatial import distance_matrix

from filterpy.stats import plot_covariance_ellipse

MAX_MEAS_RANGE = 5.0
NORM_MEAS_PROB = 0.90
num_priors = 2 # prune after each update step to this many Gaussians

class Gmm_EKF(object):
    def __init__(self, dt, T, \
                 x_initial, P_initial, alfa_initial, \
                 w_proc, Q_proc, alfa_proc, \
                 v_meas, R_meas, alfa_meas, \
                 model, 
                 num_priors):
        self.N = np.ceil( T/dt ).astype('int')
        self.dt = dt
        self.state_dim = x_initial.shape[0]
        self.meas_dim  = v_meas.shape[0]
        self.num_priors = num_priors
        # prior info for a general gaussian mixture
        self.Ngk  = x_initial.shape[1] # number of priors
        self.x    = x_initial.copy()
        self.current_output = self.x[:,0].copy()
        self.P    = P_initial.copy()
        self.alfa_prior = alfa_initial.copy()
        # info for the process noise gaussian mixture
        self.Nw   = w_proc.shape[1] # number of process noise hypotheses
        self.w    = w_proc.copy()
        self.Q    = Q_proc.copy()
        self.alfa_proc = alfa_proc.copy()
        # info for the process noise gaussian mixture
        self.Nr  = v_meas.shape[1] # number of measurement noise hypotheses
        self.v    = v_meas.copy()
        self.R    = R_meas.copy()
        self.alfa_meas = alfa_meas.copy()
        
        # for telemetry
        self.true_hidden_x = []
        self.estimated_x = []
        self.estimated_P = []
        self.estimated_alfas = []
        self.recorded_meas = []
        
        self.u = 0.0 # just a placeholder
        self.model = model
        
        self.debug = True #False
        
        self.telem = open('telemetry.csv', 'wt')
        
    def __exit__(self, exc_type, exc_value, traceback):
        print('closing the telemetry file')
        self.telem.close()
        
    def __enter__(self):
        return self
    
    def run(self, measured_ranges=None, control_inputs=None):
        # run the actual filtering + simulation (or use recording)
        # k = step number
        for k in range(self.N):
            self.time = k*self.dt
            if(self.debug):
                print('running step #%d (t=%.2f):' %(k, k*self.dt))
            # re-set the vector sizes for the new iteration
            self.x_    = np.zeros((self.state_dim, self.Ngk*self.Nw))
            self.P_    = np.zeros((self.state_dim, self.state_dim, self.Ngk*self.Nw))
            self.alfa_ = np.zeros((self.Ngk*self.Nw, 1))
            
            # could get dynamically or get all measurements a-priori
            if(measured_ranges is None):
                #import pdb; pdb.set_trace()
                self.u = self.model.control( self.current_output )
                # for debugging, zero out the controls
                #self.u = 0.
                # we're not "exposed" to the true state
                __, measured_range = self.model.step(self.u)
            else:
                # someone already got the measurements. assume, control and meas come together
                measured_range = measured_ranges[k]
                self.u = control_inputs[k]
            # predicition phase
            self.predict()
            
            # if(measured_range > 5.5):
            #     import pdb; pdb.set_trace()
            # re-set the vector sizes for the new iteration of update
            self.x__    = np.zeros((self.state_dim, self.Ngk*self.Nw*self.Nr))
            self.P__    = np.zeros((self.state_dim, self.state_dim, self.Ngk*self.Nw*self.Nr))
            self.alfa__ = np.zeros((self.Ngk*self.Nw*self.Nr, 1))
            # update phase according the the received measurement
            self.update(measured_range)
            
            # return the Nr*Nw*Ngk gaussians to Ngk by prunning states
            self.condensation()
            #self.condensation_KL_div()
            #self.condensation_max()
            
            # save the data for plotting later
            self.true_hidden_x.append(self.model.x)
            self.estimated_x.append(self.x)
            self.estimated_P.append(self.P)
            self.estimated_alfas.append(self.alfa_prior)
            self.recorded_meas.append(self.model.lidar.wall_loc_y - measured_range)
            # for now, assume the last one, the one with the highest alfa
            # will be used for state control
            self.current_output = self.x[:,-1].copy()
            
            #import pdb; pdb.set_trace()
            if(self.debug):
                print('x1_est = [%.3f; %.3f] p=%.3f*' %(self.x[0,1], self.x[1,1], self.alfa_prior[-1]))
                print('x2_est = [%.3f; %.3f] p=%.3f' %(self.x[0,0], self.x[1,0], self.alfa_prior[0]))
                nmax = np.argmax(self.alfa__)
                print('highest alfa: particle %d' %(nmax) )
                print('')
                
            if(self.model.safety_violation()):
                print('reached the wall, exitting ...')
                break
            
            # output to file
            self.write_telemetry()
            
    def predict(self):
        # predict step, to get p(x(k+1)|z(1:k))
        q = 0
        for i in range(self.Ngk):
            for j in range(self.Nw):
                x = self.model.f( self.x[:,i], self.u, self.w[:,j] )
                self.x_[:,q] = x.T
                Fq_k, __ = self.model.FJacobian_at( self.x[:,i], self.u, self.w[:,j] )
                Gq_k     = self.model.GJacobian_at( self.x[:,i], self.u, self.w[:,j] )
                self.P_[:,:,q] = np.matmul(np.matmul(Fq_k,self.P[:,:,i]), Fq_k.T) + \
                                 np.matmul(np.matmul(Gq_k,self.Q[:,:,j]), Gq_k.T)
                self.alfa_[q] = self.alfa_prior[i]*self.alfa_proc[j]
                q += 1
        # normalize the Guassian sum coefficients so it will be a valid distribution
        self.alfa_ = self.alfa_ / self.alfa_.sum()
        
    def update(self, measured_range):
        # update step to get the posterior p(x(k+1)|z(1:k+1))
        r, q = 0, 0
        I = np.eye(self.state_dim)
        for i in range(self.Ngk):
            for j in range(self.Nw):
                for l in range(self.Nr):
                    # GUY TODO: this is hardcoded to set the value of
                    # the noise of v such that the measurement would
                    # be max range on the range_max noise. need to do something more clever
                    #import pdb; pdb.set_trace()
                    # option 1: equations from class
                    #"""
                    if(l == self.Nr - 1):
                        self.v[:,l] = self.model.adjust_noise_mean(self.x_[:,q])
                        
                    Hr_k = self.model.HJacobian_at( self.x_[:,q], self.v[:,l] )
                    Pq_k = self.P_[:,:,q]
                    Rl_k = self.R[:,:,l]
                    Mq_k = self.model.MJacobian_at( self.x_[:,q], self.v[:,l] )
                    PH_k = np.matmul(Pq_k, Hr_k.T)
                    DD   = np.linalg.inv(np.matmul(Hr_k, PH_k) + \
                                         np.matmul(np.matmul(Mq_k, Rl_k), Mq_k.T))
                    Kr_k = np.matmul(PH_k, DD) # the Kalman gain
                    zhat = self.model.h(self.x_[:,q], self.v[:,l]) 
                    residual = measured_range - zhat
                    self.x__[:,r] = self.x_[:,q] + np.matmul(Kr_k, residual)
                    IKH_k = I-np.matmul(Kr_k,Hr_k) 
                    self.P__[:,:,r] = np.matmul(np.matmul(IKH_k, self.P_[:,:,q]), IKH_k.T) + \
                                      np.matmul(np.matmul(Kr_k, Rl_k), Kr_k.T)
                    # GUY TODO: is it zhat or actual z??
                    #if(measured_range > 5.):
                    #    import pdb; pdb.set_trace()
                    likelihood, __ = self.model.get_likelihood(zhat, self.x__[:,r])
                    #likelihood, __ = self.model.get_likelihood(zhat, self.x__[:,q])
                    #likelihood, __ = self.model.get_likelihood(measured_range, self.x__[:,r])
                    #self.alfa__[r] = self.alfa_[q]*likelihood # does it already incorporate the coefficient?? *self.alfa_meas[l]
                    #likelihood = self.model.get_likelihood(measured_range, self.x_[:,q])
                    #self.alfa__[r] = self.alfa_[q]*likelihood # does it already incorporate the coefficient?? *self.alfa_meas[l]
                    # this is what was written in the lecture notes
                    #likelihood = self.model.get_likelihood(zhat, self.x_[:,q])
                    self.alfa__[r] = self.alfa_[q]*self.alfa_meas[l]*likelihood
                    if(self.debug):
                        print('x(%d)+:[%.2f,%.2f,%.2f] res=%.2f likelihood=%.3f coeff=%.4f' \
                              %(r, self.x__[0,r], self.x__[1,r], self.x__[2,r], residual, likelihood, self.alfa__[r]))
                    #"""
                    # end option 1
                    # option 2: equations from paper
                    """
                    if(l == self.Nr - 1):
                        self.v[:,l] = self.model.adjust_noise_mean(self.x_[:,q])
                        
                    Hr_k = self.model.HJacobian_at( self.x_[:,q], self.v[:,l] )
                    Pq_k = self.P_[:,:,q]
                    Rl_k = self.R[:,:,l]
                    Mq_k = self.model.MJacobian_at( self.x_[:,q], self.v[:,l] )
                    PH_k = np.matmul(Pq_k, Hr_k.T)
                    DD   = np.linalg.inv(np.matmul(Hr_k, PH_k) + \
                                         np.matmul(np.matmul(Mq_k, Rl_k), Mq_k.T))
                    Kr_k = np.matmul(PH_k, DD) # the Kalman gain
                    zhat = self.model.h(self.x_[:,q], self.v[:,l]) 
                    residual = measured_range - zhat
                    self.x__[:,r] = self.x_[:,q] + np.matmul(Kr_k, residual)
                    IKH_k = I-np.matmul(Kr_k,Hr_k) 
                    self.P__[:,:,r] = np.matmul(IKH_k, self.P_[:,:,q])
                    # GUY TODO: is it zhat or actual z??
                    #if(measured_range > 5.):
                    #    import pdb; pdb.set_trace()
                    #likelihood, __ = self.model.get_likelihood(zhat, self.x__[:,r])
                    likelihood, __ = self.model.get_likelihood(zhat, self.x__[:,q])
                    #likelihood, __ = self.model.get_likelihood(measured_range, self.x__[:,r])
                    #self.alfa__[r] = self.alfa_[q]*likelihood # does it already incorporate the coefficient?? *self.alfa_meas[l]
                    #likelihood = self.model.get_likelihood(measured_range, self.x_[:,q])
                    #self.alfa__[r] = self.alfa_[q]*likelihood # does it already incorporate the coefficient?? *self.alfa_meas[l]
                    # this is what was written in the lecture notes
                    #likelihood = self.model.get_likelihood(zhat, self.x_[:,q])
                    self.alfa__[r] = self.alfa_[q]*self.alfa_meas[l]*likelihood
                    if(self.debug):
                        print('x(%d)+:[%.2f,%.2f,%.2f] res=%.2f likelihood=%.3f coeff=%.4f' \
                              %(r, self.x__[0,r], self.x__[1,r], self.x__[2,r], residual, likelihood, self.alfa__[r]))
                    """
                    # end option 2
                    r += 1
                q += 1
        # normalize the Guassian sum coefficients
        self.alfa__ /= self.alfa__.sum()
        
    # prunes the Guassians to the number the user asked for
    def condensation_max(self):
        # option1: just take highest num_priors numbers
        ind = self.alfa__.argsort(axis=0).squeeze()
        # take the last num_priors (those have the largest weights)
        combined_x    = self.x__[:,ind]
        combined_P    = self.P__[:,:,ind]
        combined_alfa = self.alfa__[ind]
        
        if(self.num_priors > len(combined_alfa)):
            # if there are less distinct states than the amount we wished for,
            # just continue with all you've got 
            self.Ngk  = len(combined_alfa)
        else:
            self.Ngk  = self.num_priors
            
        self.x    = combined_x[:,-self.Ngk:].copy()
        self.P    = combined_P[:,:,-self.Ngk:].copy()
        self.alfa_prior = combined_alfa[-self.Ngk:].copy() 
            
        self.alfa_prior /= self.alfa_prior.sum() # re-normalize
        
       
    # prunes the Guassians to the number the user asked for
    def condensation(self):
        # option1: just take highest num_priors numbers
        '''        
        ind = self.alfa__.argsort(axis=0).squeeze()
        # take the last num_priors (those have the largest weights)
        combined_x    = self.x__[:,ind]
        combined_P    = self.P__[:,:,ind]
        combined_alfa = self.alfa__[ind]
        
        # GUY TODO: at least combine Gaussians that are close and similar
        self.x    = combined_x[:,-self.num_priors:].copy()
        self.P    = combined_P[:,:,-self.num_priors:].copy()
        self.alfa_prior = combined_alfa[-self.num_priors:].copy()        
        '''
        # option2: first join close states, then take num_priors highest
        ARBITRARY_DISTANCE = 0.2
        D = distance_matrix(self.x__[:2,:].T, self.x__[:2,:].T) # only on [x,y]
        combined_x, combined_P, combined_alfa = self.x__.copy(), self.P__.copy(), self.alfa__.copy()
        merged_nodes = []
        # only merges two mixands at a time
        for i in range(D.shape[0]):
            if(i in merged_nodes):
                # already merged
                continue
            for j in range(i+1, D.shape[1]):
                if(j in merged_nodes):
                    # already merged
                    continue
                if(D[i,j] < ARBITRARY_DISTANCE):
                    # don't touch these nodes again
                    merged_nodes.append(i)
                    merged_nodes.append(j)
                    # create a single Gaussian with similar moments
                    n1, n2 = self.alfa__[i], self.alfa__[j]
                    x1, x2 = self.x__[:,i], self.x__[:,j]
                    P1, P2 = self.P__[:,:,i], self.P__[:,:,j]
                    mu = (n1*x1 + n2*x2)/(n1+n2)
                    # option 1: class
                    #P = (n1**2*P1+n2**2*P2) / \
                    #    (n1**2+n2**2) 
                    # option 2: paper
                    print('using papers formulation of condensation. ')
                    P = (n1*P1+n2*P2)/(n1+n2) + (n1*n2)/(n1+n2)**2 * np.matmul(x1-x2, (x1-x2).T)

                    # replace the first term with it, "delete" the second term
                    combined_x[:,i] = mu
                    combined_x[:,j] = combined_x[:,j]*np.nan
                    combined_P[:,:,i] = P 
                    combined_P[:,:,j] = combined_P[:,:,j]*np.nan
                    combined_alfa[i] = np.array([n1+n2])
                    combined_alfa[j] = combined_alfa[j]*np.nan
                    # since we merged, no re-joining this node with other nodes anymore
                    break
        
        # now prune
        ind  = combined_alfa.argsort(axis=0).squeeze()
        # which are marked for deletion
        mask = ~np.isnan(combined_alfa[ind]).squeeze()
        # take the last num_priors (those have the largest weights)
        combined_x    = combined_x[:,ind][:,mask]
        combined_P    = combined_P[:,:,ind][:,:,mask]
        combined_alfa = combined_alfa[ind][mask]        
        
        if(self.num_priors > len(combined_alfa)):
            # if there are less distinct states than the amount we wished for,
            # just continue with all you've got 
            self.Ngk  = len(combined_alfa)
        else:
            self.Ngk  = self.num_priors
            
        self.x    = combined_x[:,-self.Ngk:].copy()
        self.P    = combined_P[:,:,-self.Ngk:].copy()
        self.alfa_prior = combined_alfa[-self.Ngk:].copy() 
            
        self.alfa_prior /= self.alfa_prior.sum() # re-normalize

    def condensation_KL_div(self):
        combined_x, combined_P, combined_alfa = self.x__.copy(), self.P__.copy(), self.alfa__.copy()
        
        n = combined_x.shape[1]
        B = 1.*np.ones((n, n)) # so it won't find the minimum as the lower triagnle or the diagonal

        while(n > self.num_priors):        
            for i in range(n):
                for j in range(i+1, n):
                    n1, n2 = combined_alfa[i], combined_alfa[j]
                    x1, x2 = combined_x[:,i], combined_x[:,j]
                    P1, P2 = combined_P[:,:,i], combined_P[:,:,j]
                    mu = (n1*x1 + n2*x2)/(n1+n2)
                    
                    P = (n1*P1+n2*P2)/(n1+n2) + (n1*n2)/(n1+n2)**2 * np.matmul(x1-x2, (x1-x2).T)
                    B[i,j] = 0.5 * ( (n1+n2)*np.log( np.linalg.det(P) ) - \
                                     n1*np.log( np.linalg.det(P1) ) - \
                                     n2*np.log( np.linalg.det(P2) ) )
            
            lowest_ind = np.argmin(B)
            i_ind, j_ind = np.unravel_index(lowest_ind, (n,n))
            n1, n2 = combined_alfa[i_ind], combined_alfa[j_ind]
            x1, x2 = combined_x[:,i_ind], combined_x[:,j_ind]
            P1, P2 = combined_P[:,:,i_ind], combined_P[:,:,j_ind]
            mu = (n1*x1 + n2*x2)/(n1+n2)
            P = (n1*P1+n2*P2)/(n1+n2) + (n1*n2)/(n1+n2)**2 * np.matmul(x1-x2, (x1-x2).T)
            
            #import pdb; pdb.set_trace()
            combined_alfa =  np.delete(combined_alfa, [i_ind, j_ind], axis=0)
            combined_x    =  np.delete(combined_x, [i_ind, j_ind], axis=1)
            combined_P    =  np.delete(combined_P, [i_ind, j_ind], axis=2)
            
            combined_alfa =  np.append(combined_alfa, np.array([n1+n2]), axis=0)
            combined_x    =  np.append(combined_x, np.array([mu]).T, axis=1)
            combined_P    =  np.append(combined_P, np.array([P]).T, axis=2)
            
            n = combined_x.shape[1]
            B = 1.*np.ones((n, n)) # so it won't find the minimum as the lower triagnle or the diagonal
        
        # now prune
        ind  = combined_alfa.argsort(axis=0).squeeze()
        # take the last num_priors (those have the largest weights)
        combined_x    = combined_x[:,ind]
        combined_P    = combined_P[:,:,ind]
        combined_alfa = combined_alfa[ind]
        
        if(self.num_priors > len(combined_alfa)):
            # if there are less distinct states than the amount we wished for,
            # just continue with all you've got 
            self.Ngk  = len(combined_alfa)
        else:
            self.Ngk  = self.num_priors
            
        self.x    = combined_x[:,-self.Ngk:].copy()
        self.P    = combined_P[:,:,-self.Ngk:].copy()
        self.alfa_prior = combined_alfa[-self.Ngk:].copy() 
            
        self.alfa_prior /= self.alfa_prior.sum() # re-normalize
         
    # visualization        
    def plot(self):
        plot_every = 1
        for k in range(0, len(self.estimated_x), plot_every):
            # last one has the highest prob.
            for i in [self.num_priors-1]: #range(self.num_priors):
                x = self.true_hidden_x[k][0]
                y = self.true_hidden_x[k][1]
                plt.scatter(x, y, \
                        marker='.', s=40,facecolor='r')
                
                best_i = gmm.estimated_alfas[k].shape[0]
                if(best_i < self.num_priors):
                    x = self.estimated_x[k][0,best_i-1]
                    y = self.estimated_x[k][1,best_i-1]                           
                    P = self.estimated_P[k][0:2,0:2,best_i-1]
                else:
                    x = self.estimated_x[k][0,i]
                    y = self.estimated_x[k][1,i]   
                    P = self.estimated_P[k][0:2,0:2,i]
                    
                plot_covariance_ellipse((x, y), \
                                 P, std=3,
                                 facecolor='g', alpha=0.1)
                plt.scatter(x, self.recorded_meas[k], marker='x', s=40,facecolor='r')
                    
            
        plt.plot([0, x], [self.model.wall.get_y(0.), self.model.wall.get_y(x)], label='north wall', color='k', linewidth=4)
        plt.plot([0, x], [-self.model.wall.get_y(0.), -self.model.wall.get_y(x)], label='south wall', color='k', linewidth=4)
        plt.title('Robot in a corridor with Gaussian sum range noise')

    def write_telemetry(self):
        s = '%.3f, ' %self.time
        s += '%.3f, %.3f, ' %(self.model.x[0], self.model.x[1])
        for i in range(self.x.shape[1]):
            s += '%.3f, %.3f, ' %(self.x[0, i], self.x[1, i])
            s += '%.3f, %.3f, %.3f, %.3f, ' %(self.P[0, 0, i], self.P[0, 1, i], self.P[1, 0, i], self.P[1, 1, i])
            s += '%.3f, ' %(self.alfa_prior[i])
        s += '%.3f\n' %(self.recorded_meas[-1])
        
        self.telem.write(s)

# implements the dynamical equations and the gradients
class Robot(object):
    """ Simulates a robot model travelling in a corridor 
        state = [x;y;ydot]
    """
    def __init__(self, x, dt, proc_noise=np.array([1e-2, 1e-5, 1e-5])):
        self.dim_x = x.shape[0]
        self.x = x.copy() #initial estimate
        # helper class to get the noise right
        self.lidar = Lidar(a1=NORM_MEAS_PROB, a4=1.0-NORM_MEAS_PROB, norm_sig=0.3, uni_delta=0.05, \
                           meas_max=MAX_MEAS_RANGE)
        self.wall  = Wall()
        
        # true "unknown" parameters of the plant
        self.m = 1. #[Kg]
        self.const_velocity = 1. #[m/s]
        self.proc_noise = proc_noise.copy()
        self.dt = dt #[sec]
            
    def f(self, x, u, w=None):
        ''' x(k+1) = f(x(k),u(k)) + w(k);   w(k)~N(0,sigma) '''
        if(w is None):
            # if we don't send w, it means it is for simulation so sample some noise.
            # if it is sent with a vector, it means the estimator is using it
            w = self.dt * self.proc_noise * np.random.randn(self.dim_x)

        # discrete equations
        xkk    = np.zeros((self.dim_x, 1))
        # x'   = v_const
        xkk[0] = x[0] + self.const_velocity*self.dt + w[0]   
        # y'   = y'
        xkk[1] = x[1] + x[2]*self.dt + self.dt*self.dt/(2.*self.m) * u + w[1]                  
        # m*y'' = u
        xkk[2] = x[2] + self.dt/self.m * u + w[2]         # 
        
        return xkk

    def FJacobian_at(self, x, u, w):
        """ compute Jacobian of F matrix for state x,w """
        Fx = np.array([[1., 0., 0.], [0., 1., self.dt], [0., 0., 1.]])
        Fu = np.array([[0.], [(self.dt**2)/(2.*self.m)], [self.dt/self.m]])
        return Fx, Fu
    
    def GJacobian_at(self, x, u, w):
        """ compute Jacobian of G matrix for state x,w """
        #Gw = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        Gw = np.array([[self.dt, 0, 0], [0, self.dt, 0], [0, 0, self.dt]])
        return Gw

    def h(self, x, v):
        """ get noisy measurement (y direction) """
        self.lidar.wall_loc_x = x[0]
        self.lidar.wall_loc_y = self.wall.get_y(x[0])
        
        # z = Cx + v
        r = self.lidar.wall_loc_y - x[1] + v

        return r

    def HJacobian_at(self, x, v):
        """ compute Jacobian of H matrix for state x """
        #H_x = np.array([[0, -1, 0]])
        px = x[0]
        py = self.wall.get_y(x[0])
        dist = np.sqrt((px - x[0])**2 + (py - x[1])**2)
        
        H_x = np.array([[(-px + x[0])/dist,\
                         (-py + x[1])/dist,\
                         0.]])
        return H_x
    
    def MJacobian_at(self, x, v):
        """ compute Jacobian of M matrix for noise v """
        M_v = np.array([[1.]])
        return M_v

    def step(self, u):
        xk = self.x.copy()
        # update state via dynamics
        xkk = self.f(xk, u)

        # get a new measurement given the updated step
        self.lidar.wall_loc_x = xkk[0]
        self.lidar.wall_loc_y = self.wall.get_y(xkk[0])
        r = self.lidar.get_range(xkk)
        
        print('plant moved to: [%.3f; %.3f; %.3f], r=%.3f' %(xkk[0], xkk[1], xkk[2], r))
        # update the internal state
        self.x = xkk.copy()
        return xkk, r
    
    # LQR on state feedback
    def control(self, x_est):
        ref = 0. # y location
        
        Kf = np.array([0.0, 0.9171, 1.6356]) # found via matlab (lqrd)
        u = ref - np.matmul(Kf, x_est)
        
        return u
    
    # returns p(z^|x^)
    def get_likelihood(self, z, x, debug=False):
        # get a new measurement given the updated step
        self.lidar.wall_loc_x = x[0]
        self.lidar.wall_loc_y = self.wall.get_y(x[0])
        likelihood, pdf = self.lidar.get_likelihood(x)
        try:
            # because we can get measurements that are too high and then the
            # probabilities will be too low and the sum will zero out due to rounding numbers
            if(z > self.lidar.meas_max):
                z = self.lidar.meas_max
            p = likelihood(z) # of the expected measurement)
        except:
            #import pdb; pdb.set_trace()
            p = likelihood(self.lidar.meas_max) # this means we're over
        if(debug):
            x = np.linspace(0., self.lidar.meas_max, 1000)
            y = likelihood(x)
            plt.plot(x, y)
            plt.show()
        return p, pdf

    # checks if we hit a wall
    def safety_violation(self):
        w = self.wall.get_y(self.x[0])
        if(self.x[1] > w or self.x[1] < -w):
            return True
        
        return False
    
    # returns the residual v term such that given a state, the measurement is max_range
    def adjust_noise_mean(self, x):
        self.lidar.wall_loc_x = x[0]
        self.lidar.wall_loc_y = self.wall.get_y(x[0])
        # z = wall - y + v ==>
        v = self.lidar.meas_max + x[1] - self.lidar.wall_loc_y
        return v

class Wall(object):
    """ helper class to get a standard interface for the location of the wall 
    """
    def __init__(self, y=None):
        if(y is None):
            self.y = self._y_straight_line
        else:
            self.y = y

    def _y_straight_line(self, x):
        wall_loc = 3.0
        # if(1.5<x<2.):
        #     wall_loc = 6.0
        # else:
        #     wall_loc = 3.0
        return wall_loc
        
    def get_y(self, xq):
        return self.y(xq)

class Lidar(object):
    """ Simulates the Lidar signal returns from the wall. 
    """  
    def __init__(self, sensor_inst_ang=90., wall_loc=3.0, meas_max=6.0, \
                 a1=.9, a2=0.0, a3=.0, a4=.1, norm_sig=0.1, uni_delta=0.05):
        self.sensor_inst_ang = sensor_inst_ang
        self.wall_loc_x = 0. #[m]
        self.wall_loc_y = 0. #[m]
        self.meas_max = meas_max # max measurement of the range sensor [m]
        self.a1, self.a2, self.a3, self.a4, self.norm_sig, self.uni_delta = a1,a2,a3,a4,norm_sig,uni_delta
    
    # only return the distribution for the sake of getting the likelihood p,
    # does not sample
    def get_likelihood(self, x):
        # ideal_y    = np.sqrt((self.wall_loc_x-x[0])**2 + \
        #                      (self.wall_loc_y-x[1])**2)
        ideal_y    = self.wall_loc_y-x[1]
        
        # get the ideal ray to the wall
        ideal_meas = ideal_y / np.sin(np.deg2rad(self.sensor_inst_ang))
        # get the new noise distribution (because it varies with the nominal meas.)
        self.pdf, self.bins  = self.noise_dist(ideal_meas, \
                            self.a1, self.a2, self.a3, self.a4, self.norm_sig)
        # convert it to a true pdf
        self.pdf /= self.pdf.sum()
        # convert the x axis in the pdf from "measurement" to 
        # state x (y coordinate) so it will look more natural
        # to do p(z|x) outside for interp1.  (meas = wall-y)
        likelihood = interpolate.interp1d(self.bins, self.pdf)
        
        return likelihood, self.pdf
        
    def get_range(self, x):
        """ the interface to give a measurement based on the state
        """
        # add some process noise to the system
        #ideal_y    = self.wall_loc - x[1] # range = wall-y
        ideal_y    = np.sqrt((self.wall_loc_x-x[0])**2 + \
                             (self.wall_loc_y-x[1])**2)
        # get the ideal ray to the wall
        ideal_meas = ideal_y / np.sin(np.deg2rad(self.sensor_inst_ang))
        # get the new noise distribution (because it varies with the nominal meas.)
        self.pdf, self.bins  = self.noise_dist(ideal_meas, \
                            a1=self.a1, a2=self.a2, a3=self.a3, a4=self.a4, \
                            norm_sig=self.norm_sig, uni_delta=self.uni_delta)
        # only Gaussian noise (for debug)
        # pdf, bins = noise_dist(ideal_meas, a1=1.0, a2=0., a3=0., a4=0., norm_sig=0.1)
        # sample a new measurement from the inverse cdf using a uniform random number in [0,1]
        meas = self.inverse_transform_sampling(self.pdf, self.bins) 

        return meas

    # implements the noise pdf of the beam model and allows some parameters to be set
    """
    def noise_dist(self, x_true, a1=1., a2=1., a3=1., a4=.1, norm_sig=1., exp_lambda=1., uni_delta=0.15, plot=False):
        N = 100
        
       	# the discretization of the space (bins)
        # give it some extra space for the delta, too
       	x = np.linspace(0, self.meas_max + 3.*uni_delta, N)
       	# -x_true because we shift it and for some reason it looks at the truncated dist before shift :(
       	rv_norm = stats.truncnorm((0.-x_true)/norm_sig, (self.meas_max-x_true)/norm_sig, loc=x_true, scale=norm_sig) 
       	rv_exp  = stats.expon()
       	rv_uni  = stats.uniform()
       	
       	# the beam model pdf (see prob. robotics book ch. 6)
       	pdf = a1 * rv_norm.pdf(x) + \
       		  a2 * rv_exp.pdf(exp_lambda*x)*exp_lambda + \
       		  a3 * rv_uni.pdf((x-0.)/self.meas_max)/self.meas_max + \
       		  a4 * rv_uni.pdf((x-(self.meas_max-uni_delta))/uni_delta)/uni_delta
    		
        return pdf, x
    """
    # implements the noise pdf of a Guassian mixture model 
    def noise_dist(self, x_true, a1=1., a2=1., a3=1., a4=.1, \
                   norm_sig=1., exp_lambda=1., uni_delta=0.15, plot=False):
        N = 100
        
       	# the discretization of the space (bins)
        # give it some extra space for the delta, too
       	x = np.linspace(0, self.meas_max + 3.*uni_delta, N)
       	# -x_true because we shift it and for some reason it looks at the truncated dist before shift :(
       	rv_norm = stats.truncnorm((0.-x_true)/norm_sig, (self.meas_max-x_true)/norm_sig, loc=x_true, scale=norm_sig) 
       	rv_uni  = stats.truncnorm((0.-self.meas_max)/uni_delta, (self.meas_max+3.*uni_delta-self.meas_max)/uni_delta, loc=self.meas_max, scale=uni_delta) 
       	
       	# the beam model pdf (see prob. robotics book ch. 6)
       	pdf = a1 * rv_norm.pdf(x) + \
       		  a4 * rv_uni.pdf(x)
        pdf /= pdf.sum()
        return pdf, x
    
    # both creates the inverse cdf and samples and returns numbers from this distribution
    def inverse_transform_sampling(self, pdf, bin_edges, n_samples=1):
    	#import pdb; pdb.set_trace()
    	# this sort of creates the histogram by taking to adjacent pdf values and averaging them for every bin
    	pdf = 0.5 * ( pdf[:-1] + pdf[1:] )
    	# construct the CDF
    	cum_values = np.zeros(bin_edges.shape)
    	cum_values[1:] = np.cumsum(pdf*np.diff(bin_edges))
    	# normalize to a standard distribution because it wasn't done before
    	cum_values = cum_values/cum_values[-1] 
    	
    	inv_cdf = interpolate.interp1d(cum_values, bin_edges)
    	# u in [0,1]
    	u = np.random.rand(n_samples)
    	# return the function for later use
    	return inv_cdf(u)

gmm = None

if __name__ == '__main__':
    print('Gaussian Mixture Model for Extended Kalman Filter')    
    tic = timer()
    
    # set-up
    dt = 0.01 #[sec]
    T  = 5.
    
    # prior stuff for the initial guess only
    #x_initial = np.array([[0.], [0.], [0.]]) #one Gaussian for the prior
    if(num_priors == 2):
        x_initial = np.array([[0., 0.0], [0., -3.], [0., 0.]]) #one Gaussian for the prior
    else:
        x_initial = np.array([[0., 0.0, 0., 0.0,0., 0.0, 0., 0.0], \
                           [0., -3., 0., -3., 0., -3., 0., -3.], \
                           [0., 0., 0., 0., 0., 0., 0., 0.]]) #one Gaussian for the prior
    n_states, n_initial = x_initial.shape[0], x_initial.shape[1]
    P_initial = np.empty([n_states,n_states,n_initial])
    for i in range(n_initial):
        P_initial[:,:,i] = 0.31**2 * np.eye(n_states)
    alfa_initial = np.ones(n_initial)/n_initial # np.array([0.7, 0.3]) #
        
    # stuff for the process noise. in this case [wx; wy; wydot]
    w_proc = np.array([[0.], [0.], [0.]]) #one Gaussian for the distribution
    n_proc = x_initial.shape[1]
    Q_proc = np.empty([n_states,n_states,n_proc])
    for i in range(n_proc):
        Q_proc[:,:,i] = np.diag([[1e-4, 1e-5, 1e-4]])
    alfa_proc = np.ones(n_proc)/n_proc
    
    # stuff for the measurement noise. in this case Gaussian around true
    # range + Gaussian around range_max (approximation of the beam model)
    v_meas = np.array([[0.0, 3.0]]) #one Gaussian for the distribution
    n_meas, n_meas_g = v_meas.shape[0], v_meas.shape[1]
    R_meas = np.empty([n_meas,n_meas,n_meas_g])
    R_meas[:,:,0] = 0.30 * np.eye(n_meas)
    #R_meas[:,:,0] = 0.50**2 * np.eye(n_meas)
    R_meas[:,:,1] = 0.05 * np.eye(n_meas)
    #R_meas[:,:,1] = 0.05**2 * np.eye(n_meas)
    major_error = 0.9
    alfa_meas = np.array([major_error, 1.0-major_error])  # p(range max) = 0.1
    
    true_x0 = np.array([[0.0], [0.0], [0.0]])
    
    with Gmm_EKF(dt, T, \
                 x_initial, P_initial, alfa_initial, \
                 w_proc, Q_proc, alfa_proc, \
                 v_meas, R_meas, alfa_meas, \
                 Robot(true_x0, dt),
                 num_priors) as gmm:
        gmm.run()
        toc = timer()
        print('Filtering took %.3f[sec]' %(toc-tic))
        gmm.plot()
    
    
    
    

    