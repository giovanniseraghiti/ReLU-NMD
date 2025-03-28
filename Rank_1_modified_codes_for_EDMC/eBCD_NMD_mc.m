function [Theta,err,i,time]=eBCD_NMD_mc(X,r,param)

% Let d be a known threshold and Theta (usually) a low-rank matrix we compute X=max(0,dee^T-Theta),
% where we observe only the smallest values of Theta, translated by a fixed constant d.
%
% Let Omega={(i,j): X_{ij} > 0 }  and  Omega^C={(i,j): X_{ij} = 0 },
% 
% it computes an approximate solution of the following non linear matrix
% decomposition problem (NMD)
%
%       min_{Z,W,H} ||Z-(dee^T-WH)||_F^2  s.t. P_{Omega}(Z)=P_{Omega}(X) and P_{Omega^C}(Z)<=0.
%
% using an extrapolated block coordinate descent approach; 
% 
% Indirect extrapolation step: Z_alpha=alpha*Z^k+(1-alpha)*W^k*H^k,
%
% Update of W: W(alpha)=argmin_W || Z_alpha - W H^k ||_F^2, 
%
% Update of H: H(alpha)=argmin_H || Z_alpha - W(alpha) H ||_F^2
%
% Update of Z: Z(alpha)=argmin_Z || Z - W(alpha) H(alpha) ||_F^2  s.t.  max(0,Z)=X
% 
% Restarting of the momentum parameter if the residual does not decrease
%
%Accepted iterate: (W^{k+1},H^{k+1},Z^{k+1})=(W(alpha),H(alpha),Z(alpha))
%
%Otherwise:        (W^{k+1},H^{k+1},Z^{k+1})=(W^k,H^k,Z^k)

%****** Input ******
%   X              :  m-by-n matrix, sparse and non negative
%   r              :  rank of the decomposition
%   param          :  structure, containing the parameter of the model
%       .W0        =  initialization of the variable W (default: randn)
%       .H0        =  initialization of the variable H (default: randn)
%       .Z0        =  initialization of the variable Z (default: X)
%       .alpha     =  initial value for the extrapolation parameter
%       .alpha_max =  upper bound for the extrapolation parameter
%       .mu        =  increasing factor for the extrapolation parameter
%       .delta_bar =  minimum residual ratio for step acceptance
%       .check     =  if set to 1, checks the rank of (Z_alpha)H^T to detect possible rank-deficient solution,
%                     it also checks boundeness of the sequence(default=0)
%       .maxit     =  maximum number of iterations (default: 1000) 
%       .tol       =  tolerance on the relative error (default: 1.e-4)
%       .tolerr    =  tolerance on 10 successive errors (err(i+1)-err(i-10)) (default: 1.e-10)
%       .time      =  time limit (default: 60)
%       .display   =  if set to 1, it diplayes error along iterations (default: 1)
%       .observation= set to 1 if largest values are observed, set to 2 if smallest entries are observed (default=2)
%
% ****** Output ******
%   Theta   : m-by-n matrix, approximate solution of    
%             min_{Theta}||X-max(0,Theta)||_F^2  s.t. rank(Theta)=r.
%   err     : vector containing evolution of relative error along
%             iterations ||X-max(0,Theta)||_F / || X ||_F
%   i       : number of iterations  
%   time    : vector containing time counter along iterations
% 
% See the paper ''An extrapolated and provably convergent algorithm for Nonlinear Matrix Decomposition 
% with the ReLU function'', Giovanni Seraghiti, Margherita Porcelli, and Nicolas Gillis, 2024.  

[m,n]=size(X); 
 if nargin < 3
    param = [];
 end
 if ~isfield(param,'W0') || ~isfield(param,'H0')
     param.W0=randn(m,r); param.H0=randn(r,n);  
 end
 if ~isfield(param,'Z0')
    param.Z0 = X; 
end
if ~isfield(param,'maxit')
    param.maxit = 1000; 
end
if ~isfield(param,'tol')
    param.tol = 1.e-9; 
end
if ~isfield(param,'tolerr')
    param.tolerr = 1e-10;
end
if ~isfield(param,'time')
    param.time = 60; 
end
if ~isfield(param,'alpha')
    param.alpha = 1; 
end
if ~isfield(param,'alpha_max')
    param.alpha_max = 4; 
end
if ~isfield(param,'mu')
    param.mu = 0.3; 
end
if ~isfield(param,'delta_bar')
    param.delta_bar = 0.8; 
end
if ~isfield(param,'display')
    param.display = 1;
end
if ~isfield(param,'rank')
    param.check = 0;
end
if ~isfield(param,'observation')
    param.observation = 2;
end

%Detect (negative and) positive entries of X
if min(X(:)) < 0
    warnmess1 = 'The input matrix should be nonnegative. \n'; 
    warnmess2 = '         The negative entries have been set to zero.';
    warning(sprintf([warnmess1 warnmess2])); 
    X(X<0) = 0; 
end

%Compute fixed quantities for X
normX=norm(X,'fro');
idx=(X==0); idxp=(X>0);

%Create istances for the variables
Z=param.Z0; W=param.W0; H=param.H0; Theta=W*H; S=norm(Z-W*H,'fro');

%Initialize error and time counter
if param.observation==1
    err(1)=norm(Z-Theta+param.d,'fro')/normX; 
elseif param.observation==2
    err(1)=norm(Z+Theta-param.d,'fro')/normX; 
end
time(1)=0;

if param.display == 1
    disp('Running eBCD-NMD, evolution of [iteration number : relative error in %]');
end

%Display setting parameters along the iterations
cntdis = 0; numdis = 0; disp_time=0.2; alpha=param.alpha; 
for i=1:param.maxit
    tic
    %Indirect extrapolation step
    if param.observation==1
        Z_alpha=alpha*(Z)+(1-alpha)*(Theta-param.d);
    elseif param.observation==2
        Z_alpha=alpha*(Z)-(1-alpha)*(Theta-param.d);
    end

    %Update of W computing an orthogonal basis of (param.d-Z_alpha)*H^T
    if param.observation==1
       [W_e,~]=qr((Z_alpha+param.d)*H',0);
    elseif param.observation==2
       [W_e,~]=qr((-Z_alpha+param.d)*H',0);
    end    
     
    %update of H 
    if param.observation==1
       H_e=W_e'*(Z_alpha+param.d);
    elseif param.observation==2
       H_e=W_e'*(-Z_alpha+param.d);        
    end
    
    %approzimation matrix
    Theta_e=W_e*H_e;

    if param.check==1 
        if max(abs(Theta_e.*idx),[],'all')>1e10  %check on the boundness of the entries in Omega^C that needs to remain bounded to have convergence
            fprintf('The sequence might be unbounded\n')
        end
    end
    
    %Update of Z
    if param.observation==1
        Z_e=min(0,(Theta_e-param.d).*idx); 
    elseif param.observation==2
        Z_e=min(0,(-Theta_e+param.d).*idx);
    end
    Z_e=Z_e+X.*idxp;

    %Compute relative residual
    if param.observation==1
        S_e=norm(Z_e-Theta_e+param.d,'fro'); 
    elseif param.observation==2
        S_e=norm(Z_e+Theta_e-param.d,'fro'); 
    end
    res_ratio=S_e/S;  %residual ratio
    
    %Adaptive strategy to select extrapolation parameter
    if res_ratio < 1
        W=W_e; H=H_e; Z=Z_e; S=S_e; Theta=Theta_e;  
        if res_ratio > param.delta_bar
            param.mu=max(param.mu,0.25*(alpha-1)); alpha=min(alpha+param.mu,param.alpha_max);
        end
    else
        alpha=1; 
    end

    %Relative residual evaluation 
    err(i+1)=S/normX; 
    
    %Stopping criteria on relative residual
    if err(i+1)<param.tol 
        time(i+1)=time(i)+toc; %needed to have same time components as iterations
         if param.display == 1
            if mod(numdis,5) > 0, fprintf('\n'); end
            fprintf('The algorithm has converged: ||Z-WH||/||X|| < %2.0d\n',param.tol);
        end
        break
    end
    
    %Stopping criteria if the residual is not reduced sufficiently in 10 iterations
    if i >= 11  &&  abs(err(i+1) - err(i-10)) < param.tolerr
        time(i+1)=time(i)+toc; %needed to have same time components as iterations
        if param.display == 1
            if mod(numdis,5) > 0, fprintf('\n'); end
            fprintf('The algorithm has converged: rel. err.(i+1) - rel. err.(i+10) < %2.0d\n',param.tolerr);
        end
        break
    end
  
    %Stopping criteria on running time
    time(i+1)=time(i)+toc;
    if time(i+1)>param.time
        break
    end
    
    %Display the behavior of the residual along the iterations
    if param.display == 1 && time(i+1) >= cntdis 
        disp_time = min(60,disp_time*1.5);
        fprintf('[%2.2d : %2.2f] - ',i,100*err(i)); 
        cntdis = time(i+1)+disp_time; % display every disp_time
        numdis = numdis+1; 
        if mod(numdis,5) == 0
            fprintf('\n');
        end
    end
    
end
if param.display == 1
    fprintf('Final relative error: %2.2f%%, after %2.2d iterations. \n',100*err(i+1),i); 
end