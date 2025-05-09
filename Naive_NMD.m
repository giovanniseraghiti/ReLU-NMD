function [Theta,err,i,time]=Naive_NMD(X,r,param)

% Computes an approximate solution of the following non linear matrix
% decomposition problem (NMD)
%
%       min_{Z,Theta} ||Z-Theta||_F^2  s.t. rank(Theta)=r, max(0,Z)=X, 
% 
% using an simple approach that alternates between the optimal feasible Z satisfing max(0,Z)=X;
% and one optimal Theta that is the TSVD of Z, Theta=TSVD(Z,r).
% 
%****** Input ******
%   X       : m-by-n matrix, sparse and non negative
%   Theta0  : m-by-n matrix
%   r       : scalar, desired approximation rank
%   param   : structure, containing the parameter of the model
%       .Theta0 = initialization of the variable Theta (default: randn)
%       .maxit  = maximum number of iterations (default: 1000) 
%       .tol    = tolerance on the relative error (default: 1.e-9)
%       .tolerr = tolerance on 10 successive errors (err(i+1)-err(i-10)) (default: 1.e-10)
%       .time   = time limit (default: 60)
%       .display= if set to 1, it diplayes error along iterations (default: 1)
% 
% ****** Output ******
%   Theta   : m-by-n matrix, approximate solution of    
%             min_{Theta}||X-max(0,Theta)||_F^2  s.t. rank(Theta)=r.
%   err     : vector containing evolution of relative error along
%             iterations ||Z-WH||_F / || X ||_F
%   i       : number of iterations  
%   time    : vector containing time counter along iterations
% 
% See the paper ''Accelerated Algorithms for Nonlinear Matrix Decomposition 
% with the ReLU function'', Giovanni Seraghiti, Atharva Awari, Arnaud 
% Vandaele, Margherita Porcelli, and Nicolas Gillis, 2023.  
tic
[m,n]=size(X); 
 if nargin < 3
    param = [];
 end
 if ~isfield(param,'Theta0') 
     param.Theta0=randn(m,n);  
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
if ~isfield(param,'display')
    param.display = 1;
end

%Detect (negative and) positive entries of X
if min(X(:)) < 0
    warnmess1 = 'The input matrix should be nonnegative. \n'; 
    warnmess2 = '         The negative entries have been set to zero.';
    warning(sprintf([warnmess1 warnmess2])); 
    X(X<0) = 0; 
end

%Compute fix quantities for X
normX=norm(X,'fro');
idx=(X==0);
idxp=(X>0);

%Define the initial variables
Z=param.Z0; Theta=param.Theta0; 

%Initialize error and time counter
err(1)=norm(Theta-Z,'fro')/normX;
time(1)=toc;

if param.display == 1
    disp('Running Naive-NMD, evolution of [iteration number : relative error in %]');
end

%Display setting parameters along the iterations
cntdis = 0; numdis = 0; disp_time=0.5;
for i=1:param.maxit
    tic
    %Update on Z 
    Z=min(0,Theta.*idx); 
    Z=Z+X.*idxp; 

    %Update of Theta using TSVD
    [W,D,V] = tsvd(Z,r);
    H = D*V';
    Theta = W*H;          

    %Compute relative residual
    err(i+1)=norm(Z-Theta,'fro')/normX;  
        
    %Stopping condition on the relative residual
    if err(i+1)<param.tol 
        time(i+1)=time(i)+toc; %needed to have same time components as iterations
        if param.display == 1
            if mod(numdis,5) > 0, fprintf('\n'); end
            fprintf('The algorithm has converged: ||Z-Theta||/||X|| < %2.0d\n',param.tol);
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
