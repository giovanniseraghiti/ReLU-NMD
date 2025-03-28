function [Theta,err,i,time]=Naive_NMD_mc(X,r,param)

% Let d be a known threshold and Theta (usually) a low-rank matrix we compute X=max(0,dee^T-Theta),
% where we observe only the smallest values of Theta, translated by a fixed constant d.
% 
% Let Omega={(i,j): X_{ij} > 0 }  and  Omega^C={(i,j): X_{ij} = 0 },
% 
% it computes an approximate solution of the following non linear matrix
% decomposition problem (NMD)
%
%       min_{Z,W,H} ||Z-(dee^T-Theta)||_F^2  s.t. P_{Omega}(Z)=P_{Omega}(X) and P_{Omega^C}(Z)<=0.
% 
% using an simple approach that alternates between the optimal feasible Z satisfing P_{Omega}(Z)=P_{Omega}(X) and P_{Omega^C}(Z)<=0;
% and one optimal Theta that is the TSVD of Z, Theta=TSVD(Z-dee^T,r).
% 
%****** Input ******
%   X       : m-by-n matrix, sparse and non negative
%   Theta0  : m-by-n matrix
%   r       : scalar, desired approximation rank
%   param   : structure, containing the parameter of the model
%       .Theta0 =     initialization of the variable Theta (default: randn)
%       .maxit  =     maximum number of iterations (default: 1000) 
%       .tol    =     tolerance on the relative error (default: 1.e-4)
%       .tolerr =     tolerance on 10 successive errors (err(i+1)-err(i-10)) (default: 1.e-5)
%       .time   =     time limit (default: 20)
%       .display=     if set to 1, it diplayes error along iterations (default: 1)
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
% See the paper ''Accelerated Algorithms for Nonlinear Matrix Decomposition 
% with the ReLU function'', Giovanni Seraghiti, Atharva Awari, Arnaud 
% Vandaele, Margherita Porcelli, and Nicolas Gillis, 2023.  

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
    param.tol = 1.e-4; 
end
if ~isfield(param,'tolerr')
    param.tolerr = 1e-5;
end
if ~isfield(param,'time')
    param.time = 20; 
end
if ~isfield(param,'display')
    param.display = 1;
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
idx=(X==0);
idxp=(X>0);

%Create istances for variables
Z=param.Z0; Theta=param.Theta0; 

%Initialize error and time counter
if param.observation==1
    err(1)=norm(Z-Theta+param.d,'fro')/normX; 
elseif param.observation==2
    err(1)=norm(Z+Theta-param.d,'fro')/normX; 
end
time(1)=0;

if param.display == 1
    disp('Running Naive-NMD, evolution of [iteration number : relative error in %]');
end

%Display setting parameters along the iterations
cntdis = 0; numdis = 0; disp_time=0.5;
for i=1:param.maxit
    tic
    %Update on Z 
    if param.observation==1
        Z=min(0,(Theta-param.d).*idx); 
    elseif param.observation==2
        Z=min(0,(-Theta+param.d).*idx); 
    end
    Z=Z+X.*idxp; 

    %Update of Theta
    if param.observation==1
       [W,D,V] = tsvd(Z+param.d,r); 
    elseif param.observation==2
       [W,D,V] = tsvd(-Z+param.d,r);
    end
    H = D*V';
    Theta = W*H;          

    %Compute relative residual
    if param.observation==1
        err(i+1)=norm(Z-Theta+param.d,'fro')/normX;
    elseif param.observation==2
        err(i+1)=norm(Z+Theta-param.d,'fro')/normX;
    end
        
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
